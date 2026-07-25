# app/trainer/real_trainer.py
# 适配 3.9GB 显存：Qwen3-1.5B-Instruct-4bit + LoRA r=8 + seq=512 + batch=1
import os
import json
import time
import logging
from typing import Optional, Callable, Dict, Any

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset

logger = logging.getLogger("firefly.real_trainer")

# ---------- 可环境变量覆盖的超参 ----------
MODEL_PATH = os.environ.get(
    "FIREFLY_MODEL_PATH",
    "unsloth/Qwen3-1.5B-Instruct-4bit",
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
    def _load_dataset(self) -> Dataset:
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
            return load_dataset(self.data_path, split="train")
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
        logger.info(f"[RealTrainer] loading model: {MODEL_PATH}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
            load_in_4bit=True,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        raw_ds = self._load_dataset()
        fmt_ds = raw_ds.map(self._format_example)

        cfg = SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=2,
            max_steps=MAX_STEPS,
            learning_rate=LR,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_strategy="no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=fmt_ds,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
            args=cfg,
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        adapter_path = os.path.join(OUTPUT_DIR, "adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        # 转纯 safetensors（Peft 默认存 bin，fallback 用 state_dict 写）
        safetensors_path = os.path.join(OUTPUT_DIR, "lora.safetensors")
        try:
            self.model.save_pretrained(safetensors_path)
        except Exception:
            sd = self.model.state_dict()
            from safetensors.torch import save_file

            save_file(sd, safetensors_path)

        final_loss = (
            float(trainer.state.log_history[-1].get("loss", 0.0))
            if trainer.state.log_history
            else 0.0
        )

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
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "framework": "unsloth+peft+trl",
            "torch_version": torch.__version__,
            "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9
            if torch.cuda.is_available()
            else 0,
        }
        meta_path = os.path.join(OUTPUT_DIR, "firefly_trainer_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[RealTrainer] done, loss={final_loss:.4f}, elapsed={meta['elapsed_sec']}s, out={OUTPUT_DIR}"
        )
        return meta

    # ---------- 断点续跑接口（v0.2 后期接） ----------
    def resume_if_possible(self) -> bool:
        ckpt = os.path.join(CHECKPOINT_DIR, self.task_id)
        if os.path.exists(ckpt):
            logger.info(f"[RealTrainer] checkpoint found: {ckpt}")
            return True
        return False


# ── 向后兼容：task_executor.py 用的是 RealQLoRATrainer ──────────────────────
RealQLoRATrainer = RealTrainer
