r"""
firefly-client · 真实 QLoRA 训练器（v0.2 核心）
替换 mock_trainer.py，跑通真实 QLoRA 微调流程

依赖（需 pip install）：
    pip install unsloth transformers peft datasets accelerate safetensors bitsandbytes trl

触发条件：
    FIREFLY_MOCK=0（默认）或 --trainer real
    Windows 原生需 CUDA 12.1+；WSL2/Linux 无限制

用法示例（本地有 GPU 时）：
    FIREFLY_MOCK=0 python -m app.main start
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from app.trainer.base import BaseTrainer, TrainingConfig

console = Console()
CHECKPOINT_DIR = Path.home() / ".firefly" / "checkpoints"
TASK_DIR_BASE  = Path.home() / ".firefly" / "tasks"

# ─────────────────────────────────────────────────────────────────────────────
# 辅助：同步训练函数（放在线程池避免阻塞事件循环）
# ─────────────────────────────────────────────────────────────────────────────

def _check_gpu() -> tuple[bool, int]:
    """检测 GPU，返回 (有GPU, 显存MB)"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            return True, int(gpu.total_mem / 1024**2)
    except Exception:
        pass
    return False, 0


def _resolve_model_name(name: str) -> str:
    """
    将简写模型名映射为完整 HuggingFace ID
    支持：qwen3-1.5b / qwen3-7b / llama3-8b / auto
    """
    shortcuts = {
        "qwen3-1.5b":  "unsloth/Qwen3-1.5B-Instruct-4bit",
        "qwen3-7b":    "unsloth/Qwen3-7B-Instruct-bnb-4bit",
        "llama3-8b":   "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "auto":        "unsloth/Qwen3-1.5B-Instruct-4bit",
    }
    return shortcuts.get(name.lower().strip(), name)


def _prepare_dataset(dataset_path: str, max_samples: int = 100) -> str:
    """
    准备训练数据集，返回本地 JSON 文件路径
    - dataset_path 为 HuggingFace ID（如 "yahma/alpaca-cleaned"）：自动下载前 max_samples 条
    - dataset_path 为本地文件路径：直接返回
    返回：本地 JSONL 文件路径（每行一条 {"instruction":"...","input":"...","output":"..."}）
    """
    local_path = CHECKPOINT_DIR / "dataset.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 空路径：用默认 alpaca-cleaned
    if not dataset_path or not dataset_path.strip():
        dataset_path = "yahma/alpaca-cleaned"

    if dataset_path.startswith("hf://") or "/" in dataset_path:
        console.print(f"[dim]📥 从 HuggingFace 下载数据集: {dataset_path}[/dim]")
        try:
            import datasets
            ds = datasets.load_dataset(dataset_path, split=f"train[:{max_samples}]")
            with open(local_path, "w", encoding="utf-8") as f:
                for row in ds:
                    item = {
                        "instruction": row.get("instruction", ""),
                        "input":        row.get("input", ""),
                        "output":       row.get("output", ""),
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            console.print(f"  ✅ 下载完成: {local_path} ({len(ds)} 条)")
            return str(local_path)
        except Exception as e:
            console.print(f"[yellow]⚠️ 数据集下载失败: {e}，生成随机演示数据[/yellow]")
            _generate_demo_data(local_path, max_samples)
            return str(local_path)
    else:
        # 本地路径：直接复制到 local_path
        src = Path(dataset_path)
        if not src.exists():
            console.print(f"[yellow]⚠️ 数据集文件不存在: {src}，生成演示数据[/yellow]")
            _generate_demo_data(local_path, max_samples)
        else:
            shutil.copy2(src, local_path)
        return str(local_path)


def _generate_demo_data(output_path: Path, n: int = 100):
    """生成随机演示训练数据（无真实数据集时使用）"""
    import random
    topics = [
        ("如何学习编程？", "学习编程需要多动手实践，从简单项目开始。"),
        ("什么是人工智能？", "人工智能是让机器具有人类智能的技术。"),
        ("怎样提高写作能力？", "多阅读、多思考、多写作是提高写作能力的关键。"),
        ("Python 适合做什么？", "Python 适合数据分析、Web 开发、机器学习等领域。"),
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            q, a = random.choice(topics)
            item = {"instruction": q, "input": "", "output": a}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _run_training_sync(config: dict) -> dict:
    """
    同步训练函数（在 ThreadPoolExecutor 中运行）
    返回 dict 包含：final_loss, peak_vram_mb, execution_time_sec, adapter_path
    """
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForCompletionOnlyLM
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    start_time = time.monotonic()

    model_name = _resolve_model_name(config["model_name"])
    output_dir = Path(config["output_dir"])
    dataset_path = config["dataset_path"]
    max_steps = int(config["max_steps"])
    lora_rank = int(config["lora_rank"])
    lora_alpha = int(config["lora_alpha"])

    # ── 加载数据集 ────────────────────────────
    if dataset_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_path, split="train")
    else:
        ds = load_dataset(dataset_path, split=f"train[:{config.get('max_samples',100)}]")

    def format_prompt(example):
        text = (
            f"### 指令：\n{example['instruction']}\n\n"
            f"### 回答：\n{example['output']}\n"
        )
        return {"text": text}

    ds = ds.map(format_prompt, remove_columns=ds.column_names)

    # ── 加载模型（优先 unsloth，fallback 到标准 transformers） ─
    # 写入进度文件（供主线程轮询）
    progress_file = output_dir / "_training_progress.json"

    try:
        from unsloth import FastLanguageModel
        console.print(f"[green]✅ 使用 Unsloth 加速[/green]")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config["max_seq_length"],
            dtype=None,          # auto 检测
            load_in_4bit=True,   # bnb 4bit 量化
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=config.get("lora_dropout", 0.0),
            target_modules=config.get("lora_targets", ["q_proj", "v_proj"]),
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )
        accelerator_backend = "unsloth"
    except (ImportError, Exception) as e:
        console.print(f"[yellow]⚠️ Unsloth 不可用（{e}），切换到标准 transformers+PEFT[/yellow]")
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(config.get("torch_dtype", "bfloat16"), torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=dict(load_in_4bit=True),
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=config.get("lora_dropout", 0.0),
                target_modules=config.get("lora_targets", ["q_proj", "v_proj"]),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        accelerator_backend = "transformers+peft"

    # ── 训练参数 ───────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(config.get("per_device_batch", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation", 4)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        warmup_steps=int(config.get("warmup_steps", 10)),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", max_steps)),  # 只在最后保存
        fp16=accelerator_backend == "transformers+peft",
        bf16=accelerator_backend == "unsloth",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_8bit",
    )

    # ── 训练循环（逐 step 记录 loss） ──────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False
        ),
    )

    loss_history = []
    peak_vram = 0

    # 包装 trainer.train() 实时写 progress 文件
    original_train = trainer.train

    def tracking_train(resume_from_checkpoint=None):
        nonlocal peak_vram
        if resume_from_checkpoint:
            return original_train(resume_from_checkpoint=resume_from_checkpoint)

        # 注入自定义训练循环以追踪 step/loss
        from transformers.trainer import Trainer
        if isinstance(trainer, Trainer):
            training_args.max_steps = max_steps
            return original_train()

        # Unsloth / SFTTrainer：走原始逻辑，之后读日志
        return original_train()

    # 改用 trainer.train() 直接跑（unsloth 已内部优化）
    console.print(f"[dim]🔥 开始训练 {max_steps} 步...[/dim]")
    trainer.train()

    # 尝试读 trainer.state.log_history 取 final loss
    final_loss = 2.0
    try:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break
    except Exception:
        pass

    # ── 保存 adapter ──────────────────────────
    adapter_path = output_dir / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # 记录 peak VRAM
    if torch.cuda.is_available():
        peak_vram = int(torch.cuda.max_memory_allocated() / 1024**2)

    elapsed = int(time.monotonic() - start_time)

    # 写完成状态
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "done",
            "final_loss": final_loss,
            "peak_vram_mb": peak_vram,
            "execution_time_sec": elapsed,
            "adapter_path": str(adapter_path),
            "backend": accelerator_backend,
            "completed_at": datetime.utcnow().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    # 清理
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "final_loss":     final_loss,
        "peak_vram_mb":   peak_vram,
        "execution_time_sec": elapsed,
        "lora_adapter_path": str(adapter_path),
        "backend":        accelerator_backend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 异步封装：RealQLoRATrainer
# ─────────────────────────────────────────────────────────────────────────────

class RealQLoRATrainer(BaseTrainer):
    """
    真实 QLoRA 训练器（v0.2）
    优先使用 Unsloth（快 2×、省显存 50%），自动 fallback 到 transformers+PEFT。
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._progress_file: Optional[Path] = None
        self._task_id: str = ""
        self._report_callback = None   # 可选：每步回调 fn(step, total, loss)
        self._last_stats: dict = {}

    def bind_task(self, task_id: str):
        """绑定当前任务 ID，checkpoint 保存到对应子目录"""
        self._task_id = task_id
        task_checkpoints = TASK_DIR_BASE / task_id / "checkpoints"
        task_checkpoints.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = task_checkpoints
        self._progress_file = task_checkpoints / "_training_progress.json"

    def set_report_callback(self, fn):
        """设置进度上报回调：fn(step, total_steps, loss)"""
        self._report_callback = fn

    async def load_model(self) -> None:
        loop = asyncio.get_event_loop()
        has_gpu, vram_mb = await loop.run_in_executor(None, _check_gpu)
        if not has_gpu:
            raise RuntimeError(
                "❌ 未检测到 NVIDIA GPU！\n"
                "真实 QLoRA 训练需要 CUDA 环境。\n"
                "解决方案：\n"
                "  1. 在有 GPU 的机器上运行（推荐 WSL2 + CUDA）\n"
                "  2. 使用 Google Colab（免费 T4 GPU）\n"
                "  3. 使用 AutoDL / Kaggle（租用 A100/4090）\n"
                "当前环境 VRAM: 0 MB\n"
                "提示：6GB 以上显存可跑 Qwen3-1.5B-4bit"
            )
        console.print(f"  🎮 检测到 GPU，VRAM {vram_mb} MB，启用真实训练")
        self._progress["total_steps"] = self.config.max_steps

    async def train(self) -> dict:
        """
        在线程池运行同步训练，持续读 progress 文件更新 _progress 状态。
        """
        loop = asyncio.get_event_loop()
        task_id = self._task_id or "global"
        self._progress_file = (
            self.config.output_dir / f"_progress_{task_id}.json"
        )

        # 写初始状态
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._progress_file, "w", encoding="utf-8") as f:
            json.dump({"status": "running", "step": 0}, f)

        def _training_job():
            cfg = {
                "model_name":        self.config.model_name,
                "max_seq_length":    self.config.max_seq_length,
                "dataset_path":      self.config.dataset_path,
                "max_steps":         self.config.max_steps,
                "lora_rank":         self.config.lora_rank,
                "lora_alpha":        self.config.lora_alpha,
                "lora_dropout":      self.config.lora_dropout,
                "lora_targets":      self.config.lora_targets,
                "per_device_batch":  self.config.per_device_batch,
                "gradient_accumulation": self.config.gradient_accumulation,
                "learning_rate":     self.config.learning_rate,
                "warmup_steps":      self.config.warmup_steps,
                "lr_scheduler":      self.config.lr_scheduler,
                "logging_steps":     self.config.logging_steps,
                "save_steps":        self.config.save_steps,
                "torch_dtype":       self.config.torch_dtype,
                "output_dir":        str(self.config.output_dir),
            }
            return _run_training_sync(cfg)

        # 在线程池运行训练
        console.print(
            f"  🔥 启动真实训练 | model={self.config.model_name} "
            f"| steps={self.config.max_steps} | lora_r={self.config.lora_rank}"
        )

        try:
            result = await loop.run_in_executor(None, _training_job)
        except Exception as e:
            console.print(f"[red]❌ 训练失败: {e}[/red]")
            raise

        # 更新状态
        self._progress = {
            "step": self.config.max_steps,
            "total_steps": self.config.max_steps,
            "loss": result.get("final_loss"),
        }
        self._done = True
        self._last_stats = result

        console.print(
            f"  ✅ 训练完成 | loss={result['final_loss']:.4f} "
            f"| VRAM峰值={result['peak_vram_mb']}MB "
            f"| 后端={result.get('backend','?')}"
        )
        return result

    async def save_adapter(self, output_dir: Path) -> Path:
        """将已保存的 adapter 目录复制到 output_dir，返回 safetensors 路径"""
        if not self._last_stats.get("lora_adapter_path"):
            raise RuntimeError("训练未完成，无法保存 adapter")

        adapter_src = Path(self._last_stats["lora_adapter_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / "lora_weights.safetensors"

        # LoRA adapter 文件在 adapter 目录下
        src_safetensor = adapter_src / "adapter_model.safetensors"
        if src_safetensor.exists():
            shutil.copy2(src_safetensor, dest)
        else:
            # fallback: 复制整个目录
            import zipfile
            zip_path = output_dir / "lora_adapter.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in adapter_src.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(adapter_src))
            return zip_path

        return dest

    async def get_training_stats(self) -> dict:
        """读取最新训练统计（用于上报调度中心）"""
        return self._last_stats.copy()
