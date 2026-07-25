"""
firefly-client · 训练器基类
定义统一接口，所有训练器（mock / real）均实现此接口
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Optional


@dataclass
class TrainingConfig:
    """
    训练配置（传递给 BaseTrainer 子类）
    v0.2 默认值针对 unsloth/Qwen3-1.5B-Instruct-4bit（小显存友好）
    """
    # ── 模型 ───────────────────────────────
    model_name: str = "unsloth/Qwen3-1.5B-Instruct-4bit"
    max_seq_length: int = 2048

    # ── LoRA ────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # LoRA 目标模块（q_proj+v_proj 最省显存）
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ── 训练 ───────────────────────────────
    dataset_path: str = ""          # 本地 JSON 文件，或 HuggingFace ID
    max_steps: int = 100
    learning_rate: float = 2e-4
    per_device_batch: int = 1
    gradient_accumulation: int = 4   # effective batch = 4
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50

    # ── 硬件 ───────────────────────────────
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"

    # ── 输出 ────────────────────────────────
    output_dir: Path = field(
        default_factory=lambda: Path.home() / ".firefly" / "checkpoints"
    )


class BaseTrainer(abc.ABC):
    """
    训练器抽象基类

    所有子类必须实现：
      - load_model()      加载模型
      - train()           执行训练，返回训练统计
      - save_adapter()    保存 LoRA adapter
      - get_progress()    返回 {step, total_steps, loss}
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._progress: dict = {"step": 0, "total_steps": 0, "loss": None}
        self._done = False

    @abc.abstractmethod
    async def load_model(self) -> None:
        """异步加载模型和 tokenizer"""
        raise NotImplementedError

    @abc.abstractmethod
    async def train(self) -> dict:
        """
        执行完整训练流程
        返回训练统计 dict，包含：
          - final_loss: float
          - peak_vram_mb: int
          - execution_time_sec: int
          - lora_adapter_path: str
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def save_adapter(self, output_dir: Path) -> Path:
        """保存 LoRA adapter 到 output_dir，返回 safetensors 路径"""
        raise NotImplementedError

    def get_progress(self) -> dict:
        """返回当前进度 {step, total_steps, loss}"""
        return self._progress.copy()

    @property
    def is_done(self) -> bool:
        return self._done
