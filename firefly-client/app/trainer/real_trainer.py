# app/trainer/real_trainer.py
# 适配 3.9GB 显存：Qwen3-1.5B-Instruct-4bit + LoRA r=8 + seq=512 + batch=1
# NOTE: 不使用 unsloth，直接用 transformers+peft+trl（torch>=2.0 均可）
from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional, Callable, Dict, Any

import torch

logger = logging.getLogger("firefly.real_trainer")

# ---------- 可环境变量覆盖的超参 ----------
MODEL_PATH = os.environ.get(
    "FIREFLY_MODEL_PATH",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
LORA_R = int(os.environ.get("FIREFLY_LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("FIREFLY_LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.environ.get("FIREFLY_LORA_DROPOUT", "0.05"))
MAX_SEQ_LENGTH = int(os.environ.get("FIREFLY_MAX_SEQ_LENGTH", "512"))
PER_DEVICE_BATCH = int(os.environ.get("FIREFLY_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("FIREFLY_GRAD_ACCUM", "4"))
LR = float(os.environ.get("FIREFLY_LR", "2e-4"))
MAX_STEPS = int(os.environ.get("FIREFLY_MAX_STEPS", "60"))
OUTPUT_DIR = os.environ.get(
    "FIREFLY_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), ".firefly", "train_output"),
)
CHECKPOINT_DIR = os.environ.get(
    "FIREFLY_CHECKPOINT_DIR",
    os.path.join(os.path.expanduser("~"), ".firefly", "checkpoints"),
)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # 3.9GB 只挂 q,v，省显存


def _is_bf16_supported() -> bool:
    """检查当前 GPU 是否支持 bfloat16."""
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


class RealTrainer:
    def __init__(
        self,
        task_id: str,
        data_path: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.task_id = task_id
        self.data_path = data_path
        self.progress_callback = progress_callback
        self.model = None
        self.tokenizer = None
        self.start_time = time.time()

    # ---------- 数据加载（三态） ----------
    def _load_dataset(self):
        # 延迟导入 datasets
        from datasets import Dataset
        # 1) 本地 JSONL
        if self.data_path and self.data_path.endswith(".jsonl"):
            return Dataset.from_json(self.data_path)
        # 2) 本地 JSON（list of dict）
        if self.data_path and self.data_path.endswith(".json"):
            with open(self.data_path, encoding="utf-8") as f:
                rows = json.load(f)
            return Dataset.from_list(rows)
        # 3) HuggingFace dataset 名
        if self.data_path and "/" in self.data_path and not os.path.exists(self.data_path):
            from datasets import load_dataset as ld
            return ld(self.data_path, split="train")
        # 4) Demo 冒烟数据
        demo = [
            {
                "instruction": "解释什么是 LoRA",
                "output": "LoRA 是低秩适配，冻结原权重只训小矩阵。",
            },
            {
                "instruction": "用一句话介绍 Qwen3",
                "output": "Qwen3 是通义千问第三代开源大模型系列。",
            },
            {
                "instruction": "法律问答应注意什么",
                "output": "应标注'不构成法律意见'，并建议咨询执业律师。",
            },
            {
                "instruction": "Python 怎么读文件",
                "output": "用 open('a.txt', encoding='utf-8').read()。",
            },
        ]
        return Dataset.from_list(demo)

    @staticmethod
    def _format_example(ex):
        instr = ex.get("instruction", ex.get("prompt", ""))
        out = ex.get("output", ex.get("response", ""))
        return {
            "text": (
                "<|im_start|>system\n你是一个有帮助的助手<|im_end|>\n"
                f"<|im_start|>user\n{instr}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>"
            )
        }

    # ---------- 训练主流程 ----------
    def train(self) -> Dict[str, Any]:
        # 延迟导入 heavy 库
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig

        bf16 = _is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        logger.info(f"[RealTrainer] loading model: {MODEL_PATH} (bf16={bf16})")

        # 尝试 4-bit 量化（bitsandbytes 可能在部分 torch 版本有兼容性问题）
        # 如果失败，改用 float16 直接加载（24GB 显存足够容纳 Qwen2.5-1.5B）
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("[RealTrainer] Model loaded with 4-bit quantization")
        except (AttributeError, Exception) as e:
            logger.warning(f"[RealTrainer] 4-bit quantization failed ({e}), "
                          f"falling back to float16 direct load (24GB VRAM sufficient)")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=dtype,
                device_map="cuda",
                trust_remote_code=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"

        # LoRA 配置
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=None,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        logger.info(f"[RealTrainer] trainable params: "
                    f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        raw_ds = self._load_dataset()
        # trl 0.24: use formatting_func (receives dict -> returns str)
        def _fmt(ex: dict) -> str:
            instr = ex.get("instruction", ex.get("prompt", ""))
            out = ex.get("output", ex.get("response", ""))
            return (
                "<|im_start|>system\n你是一个有帮助的助手<|im_end|>\n"
                f"<|im_start|>user\n{instr}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>"
            )

        # max_seq_length goes into SFTConfig for trl >= 0.9
        cfg = SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=2,
            max_steps=MAX_STEPS,
            max_length=MAX_SEQ_LENGTH,
            learning_rate=LR,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_strategy="no",
            report_to="none",
        )

        # trl 0.24: formatting_func + processing_class
        # max_length is set in SFTConfig (not max_seq_length)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=raw_ds,
            args=cfg,
            processing_class=self.tokenizer,
            formatting_func=_fmt,
        )

        # 进度回调（注入 trainer.log）
        orig_log = trainer.log

        def _patched_log(logs, *a, **k):
            if self.progress_callback:
                self.progress_callback(
                    {
                        "task_id": self.task_id,
                        "step": int(logs.get("step", 0)),
                        "loss": float(logs.get("loss", 0.0)),
                        "lr": float(logs.get("learning_rate", 0.0)),
                        "elapsed": time.time() - self.start_time,
                    }
                )
            return orig_log(logs, *a, **k)

        trainer.log = _patched_log

        logger.info("[RealTrainer] start training")
        trainer.train()

        # ---------- 导出 safetensors + meta ----------
        run_dir = os.path.join(OUTPUT_DIR, self.task_id)
        os.makedirs(run_dir, exist_ok=True)
        adapter_path = os.path.join(run_dir, "adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        # 转纯 safetensors（只含 LoRA trainable params，不含 frozen base）
        safetensors_path = os.path.join(run_dir, "lora.safetensors")
        from safetensors.torch import save_file, load_file
        adapter_sf = os.path.join(adapter_path, "adapter_model.safetensors")
        sd = load_file(adapter_sf)
        save_file(sd, safetensors_path)

        final_loss = 0.0
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry and entry["loss"] and entry["loss"] > 0:
                final_loss = float(entry["loss"])
                break

        # vram_gb: handle torch 2.5 renamed attribute (total_mem -> total_memory)
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        except AttributeError:
            try:
                vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            except AttributeError:
                vram = 0.0

        meta = {
            "task_id": self.task_id,
            "model_base": MODEL_PATH,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "max_seq_length": MAX_SEQ_LENGTH,
            "per_device_batch_size": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "max_steps": MAX_STEPS,
            "final_loss": final_loss,
            "adapter_path": safetensors_path,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "framework": "transformers+peft+trl",
            "torch_version": torch.__version__,
            "vram_gb": vram,
        }

        meta_path = os.path.join(run_dir, "firefly_trainer_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[RealTrainer] done, loss={final_loss:.4f}, elapsed={meta['elapsed_sec']}s"
        )
        return meta

    # ---------- 断点续跑接口 ----------
    def resume_if_possible(self) -> bool:
        ckpt = os.path.join(CHECKPOINT_DIR, self.task_id)
        if os.path.exists(ckpt):
            logger.info(f"[RealTrainer] checkpoint found: {ckpt}")
            return True
        return False


# 向后兼容别名
RealQLoRATrainer = RealTrainer
