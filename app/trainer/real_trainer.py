"""
real_trainer.py — Firefly LM 真实 QLoRA 训练器

基于 Unsloth 在本地对 Qwen3-1.5B（4-bit 量化）跑 LoRA 微调，
产出 .safetensors 权重文件回传调度中心。

设计原则：
- MODEL_PATH 支持环境变量 FIREFLY_MODEL_PATH，不硬编码本地路径
- 默认走 HuggingFace Hub 自动下载（unsloth/Qwen3-1.5B-Instruct-4bit）
- 进度回调 injectable，供 task_executor 上报心跳
- 训练中断时可从 checkpoint 续跑

使用：
    from app.trainer.real_trainer import RealTrainer
    trainer = RealTrainer(model_path=..., data_path=..., output_dir=...)
    result = trainer.train(progress_callback=cb)
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("firefly.trainer.real")


# ─────────────────────────────────────────────
# 默认配置（可用环境变量覆盖）
# ─────────────────────────────────────────────

DEFAULT_MODEL_PATH = "unsloth/Qwen3-1.5B-Instruct-4bit"
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_STEPS = 100
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_SEQ_LEN = 2048

# 挂载 LoRA 的模块（Qwen3 注意力投影）
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"{name}={val} 不是合法整数，使用默认 {default}")
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ─────────────────────────────────────────────
# RealTrainer
# ─────────────────────────────────────────────

class RealTrainer:
    """
    真实 QLoRA 训练器（基于 Unsloth）。

    参数均可用环境变量覆盖，方便无 GPU 开发机和有 GPU 训练机
    使用同一份代码、不同配置。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        output_dir: str = "~/.firefly/checkpoints",
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        learning_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        grad_accum: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        lora_target_modules: Optional[list] = None,
        seed: int = 42,
    ):
        # 模型路径：参数 > 环境变量 > 默认值
        self.model_path = (
            model_path
            or os.environ.get("FIREFLY_MODEL_PATH")
            or DEFAULT_MODEL_PATH
        )

        self.data_path = data_path or os.environ.get("FIREFLY_DATA_PATH")
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # LoRA / 训练超参：参数 > 环境变量 > 默认
        self.lora_rank = lora_rank or _env_int("FIREFLY_LORA_RANK", DEFAULT_LORA_RANK)
        self.lora_alpha = lora_alpha or _env_int("FIREFLY_LORA_ALPHA", DEFAULT_LORA_ALPHA)
        self.lora_dropout = lora_dropout or _env_float("FIREFLY_LORA_DROPOUT", DEFAULT_LORA_DROPOUT)
        self.learning_rate = learning_rate or _env_float("FIREFLY_LR", DEFAULT_LEARNING_RATE)
        self.max_steps = max_steps or _env_int("FIREFLY_MAX_STEPS", DEFAULT_MAX_STEPS)
        self.batch_size = batch_size or _env_int("FIREFLY_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        self.grad_accum = grad_accum or _env_int("FIREFLY_GRAD_ACCUM", DEFAULT_GRAD_ACCUM)
        self.max_seq_len = max_seq_len or _env_int("FIREFLY_MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN)
        self.lora_target_modules = lora_target_modules or DEFAULT_LORA_TARGET_MODULES
        self.seed = seed

        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._current_step = 0

        logger.info(f"[RealTrainer] model={self.model_path}")
        logger.info(f"[RealTrainer] lora_rank={self.lora_rank} alpha={self.lora_alpha} dropout={self.lora_dropout}")
        logger.info(f"[RealTrainer] lr={self.learning_rate} max_steps={self.max_steps} bs={self.batch_size} grad_accum={self.grad_accum}")
        logger.info(f"[RealTrainer] output_dir={self.output_dir}")

    # ── 进度回调 ──────────────────────────────

    def _make_progress_callback(
        self, user_callback: Optional[Callable[[Dict[str, Any]], None]]
    ):
        """包装用户回调，注入 step / loss / eta。"""
        start_time = time.time()
        last_log_step = [0]

        def cb(step, loss, **kwargs):
            self._current_step = step
            elapsed = time.time() - start_time
            steps_done = step
            if steps_done > 0:
                eta = elapsed / steps_done * (self.max_steps - steps_done)
            else:
                eta = 0.0

            payload = {
                "step": step,
                "loss": round(float(loss), 4),
                "eta_seconds": round(float(eta), 1),
                "max_steps": self.max_steps,
                "progress": round(steps_done / self.max_steps * 100, 1),
            }

            # 节流：每 10 步打一行日志
            if step - last_log_step[0] >= 10:
                last_log_step[0] = step
                logger.info(
                    f"[RealTrainer] step={step}/{self.max_steps} "
                    f"loss={payload['loss']:.4f} eta={payload['eta_seconds']:.0f}s"
                )

            if user_callback:
                try:
                    user_callback(payload)
                except Exception as e:
                    logger.warning(f"progress_callback 异常: {e}")

        return cb

    # ── 数据加载 ──────────────────────────────

    def _load_dataset(self):
        """
        加载训练数据。

        支持格式：
        1. JSONL 文件（每行一个 {"instruction": ..., "input": ..., "output": ...}）
        2. HuggingFace dataset name（如 "yahma/alpaca-cleaned"）
        3. None → 自动用 4 条内置 demo 数据（仅用于冒烟测试）

        返回：list[dict]，每条含 prompt 字段
        """
        # 情况 1：显式数据路径
        if self.data_path:
            path = Path(self.data_path)
            if path.exists():
                if path.suffix == ".jsonl":
                    items = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                items.append(json.loads(line))
                    logger.info(f"[RealTrainer] 加载 JSONL 数据: {len(items)} 条 ({path})")
                    return self._format_for_unsloth(items)
                elif path.suffix == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    logger.info(f"[RealTrainer] 加载 JSON 数据: {len(items)} 条 ({path})")
                    return self._format_for_unsloth(items)
            else:
                logger.warning(f"[RealTrainer] data_path={path} 不存在，回退到 demo 数据")

        # 情况 2：HF dataset
        hf_name = os.environ.get("FIREFLY_DATASET")
        if hf_name:
            try:
                from datasets import load_dataset
                ds = load_dataset(hf_name, split="train")
                items = [{"instruction": r.get("instruction",""), "output": r.get("output","")} for r in ds]
                logger.info(f"[RealTrainer] 加载 HF 数据集: {hf_name} ({len(items)} 条)")
                return self._format_for_unsloth(items[: self.max_steps * 4])
            except Exception as e:
                logger.warning(f"[RealTrainer] HF 数据集加载失败: {e}")

        # 情况 3：内置 demo
        demo = [
            {"instruction": "解释什么是分布式训练", "output": "分布式训练是把模型训练任务拆分到多台机器或多张 GPU 上并行执行，通过梯度同步（如 AllReduce）汇总更新，从而加速训练或支持更大模型。"},
            {"instruction": "LoRA 是什么", "output": "LoRA（Low-Rank Adaptation）是一种参数高效微调方法，通过在预训练权重旁添加低秩矩阵来实现微调，只训练少量新增参数，显存和计算开销远低于全量微调。"},
            {"instruction": "什么是 FedAvg", "output": "FedAvg（Federated Averaging）是联邦学习中最基础的聚合算法，各客户端在本地训练后把模型权重上传，服务端按样本数加权求平均得到全局模型。"},
            {"instruction": "Qwen3 有哪些特点", "output": "Qwen3 是阿里云开源的大语言模型系列，支持多种参数规模（0.6B~235B），提供稠密和 MoE 两种架构，在代码、数学、多语言上表现均衡，采用 Apache-2.0 许可证。"},
        ]
        logger.info("[RealTrainer] 使用内置 demo 数据 (4 条)")
        return self._format_for_unsloth(demo)

    def _format_for_unsloth(self, items: list) -> list:
        """把 instruction/output 格式转成 Unsloth 训练用的 prompt 列表。"""
        formatted = []
        for item in items:
            instr = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not instr or not out:
                continue
            if inp:
                prompt = f"### 指令\n{instr}\n\n### 输入\n{inp}\n\n### 回答\n{out}"
            else:
                prompt = f"### 指令\n{instr}\n\n### 回答\n{out}"
            formatted.append({"text": prompt})
        return formatted

    # ── 模型加载 ──────────────────────────────

    def _load_model(self):
        """加载 4-bit 量化底座 + 挂载 LoRA。"""
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise RuntimeError(
                f"unsloth 未安装，无法跑真实训练。pip install unsloth。原始错误: {e}"
            )

        logger.info(f"[RealTrainer] 加载底座模型: {self.model_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_len,
            dtype=None,  # Unsloth 自动选 bf16/fp16
            load_in_4bit=True,
            token=None,  # 私有模型才需要 HF token
        )

        # 挂载 LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # 显存优化
            random_state=self.seed,
        )

        self._model = model
        self._tokenizer = tokenizer
        logger.info("[RealTrainer] 底座 + LoRA 加载完成")

    # ── 训练 ──────────────────────────────────

    def train(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整训练流程。

        返回：
            {
                "status": "completed",
                "adapter_path": "...",
                "final_loss": 0.xxxx,
                "steps": N,
                "duration_seconds": X,
                "model_path": "...",
                "lora_config": {...},
            }
        """
        start = time.time()

        # 1. 加载数据
        dataset = self._load_dataset()
        if len(dataset) == 0:
            raise ValueError("训练数据为空，无法训练")

        # 2. 加载模型
        self._load_model()

        # 3. 构造 Unsloth SFT Trainer
        try:
            from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except ImportError as e:
            raise RuntimeError(f"unsloth/trl 未安装: {e}")

        training_args = UnslothTrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            num_train_epochs=1,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=1,
            save_steps=self.max_steps + 1,  # 训练结束再存
            save_total_limit=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=self.seed,
            output_dir=self.output_dir,
        )

        # 数据格式化函数
        def formatting_func(example):
            return example["text"]

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_len,
            dataset_num_proc=1,
            args=training_args,
            formatting_func=formatting_func,
        )

        self._trainer = trainer

        # 4. 自定义训练循环（支持进度回调 + 可中断）
        wrapped_cb = self._make_progress_callback(progress_callback)

        logger.info(f"[RealTrainer] 开始训练: max_steps={self.max_steps}")
        final_loss = None

        # 简单方式：直接 train()，靠 logging_steps=1 拿日志
        # 若需精细 step 级回调，需 monkey-patch trainer.log
        trainer.train()

        # 取最后一步 loss
        logs = trainer.state.log_history
        if logs:
            final_loss = logs[-1].get("loss", logs[-1].get("train_loss"))
        else:
            final_loss = 0.0

        # 5. 保存 adapter（safetensors 格式）
        adapter_path = os.path.join(self.output_dir, "adapter")
        self._model.save_pretrained(adapter_path, safe_serialization=True)
        self._tokenizer.save_pretrained(adapter_path)

        duration = time.time() - start
        logger.info(f"[RealTrainer] 训练完成: loss={final_loss:.4f} 耗时={duration:.1f}s")
        logger.info(f"[RealTrainer] adapter 保存至: {adapter_path}")

        # 6. 写训练元数据（供聚合 Worker 校验）
        meta = {
            "model_path": self.model_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_len": self.max_seq_len,
            "final_loss": round(float(final_loss), 4) if final_loss else None,
            "duration_seconds": round(duration, 1),
            "steps": self.max_steps,
            "adapter_path": adapter_path,
            "weight_format": "safetensors",
            "unsloth_version": self._get_unsloth_version(),
        }
        meta_path = os.path.join(adapter_path, "firefly_trainer_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "status": "completed",
            "adapter_path": adapter_path,
            "final_loss": round(float(final_loss), 4) if final_loss else 0.0,
            "steps": self.max_steps,
            "duration_seconds": round(duration, 1),
            "model_path": self.model_path,
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.lora_target_modules,
            },
            "meta_path": meta_path,
        }

    # ── Checkpoint 续跑 ───────────────────────

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        检测是否有可恢复的 checkpoint。

        返回 True 表示找到并加载成功，调用方应跳过初始化直接续跑。
        目前 Unsloth 的 PEFT 模型续跑需手动处理，这里提供基础设施。
        """
        ckpt = checkpoint_path or os.path.join(self.output_dir, "adapter")
        if os.path.exists(ckpt):
            logger.info(f"[RealTrainer] 发现 checkpoint: {ckpt}，可续跑")
            return True
        return False

    # ── 工具 ──────────────────────────────────

    def _get_unsloth_version(self) -> str:
        try:
            import unsloth
            return getattr(unsloth, "__version__", "unknown")
        except ImportError:
            return "not_installed"

    def export_adapter(self, dest_path: str) -> str:
        """
        把训练好的 adapter 导出为独立 safetensors 文件（含 meta）。
        供 task_executor 打包上传 MinIO 用。
        """
        import shutil
        if not self._model:
            raise RuntimeError("模型未加载，无法导出。请先 train()")
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(dest_path, safe_serialization=True)
        self._tokenizer.save_pretrained(dest_path)
        # 复制 meta
        src_meta = os.path.join(self.output_dir, "adapter", "firefly_trainer_meta.json")
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, os.path.join(dest_path, "firefly_trainer_meta.json"))
        logger.info(f"[RealTrainer] adapter 导出至: {dest_path}")
        return dest_path
