"""
firefly-client · Mock 训练器（v0.1 保留，向后兼容）
触发条件：FIREFLY_MOCK=1 或 --mock 时使用
"""
import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors.numpy import save_file as st_save

from app.trainer.base import BaseTrainer, TrainingConfig


class MockTrainer(BaseTrainer):
    """
    纯模拟训练器，仅用于：
    - 调度中心联调（无 GPU 机器）
    - E2E 测试（CI/CD）
    - UI 流程验证
    不产出真实权重，不参与 FedAvg 聚合。
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._step = 0
        self._total_steps = config.max_steps
        self._loss = 3.5

    async def load_model(self) -> None:
        # 无 GPU 环境跳过加载
        self._progress["total_steps"] = self._total_steps

    async def train(self) -> dict:
        import os
        e2e = os.environ.get("FIREFLY_E2E") == "1"
        total = 5 if e2e else self._total_steps
        sleep_t = 0.05 if e2e else 0.3

        start = time.monotonic()

        for step in range(total):
            await asyncio.sleep(sleep_t)
            self._step = step + 1
            self._loss = max(0.8, 3.5 - (self._step / self._total_steps) * 2.7)
            self._progress = {
                "step": self._step,
                "total_steps": self._total_steps,
                "loss": round(self._loss, 4),
            }

        elapsed = int(time.monotonic() - start)
        self._done = True
        return {
            "final_loss": round(self._loss, 4),
            "peak_vram_mb": 0,          # mock 无 GPU
            "execution_time_sec": elapsed,
            "lora_adapter_path": "",     # mock 不产出真实权重
        }

    async def save_adapter(self, output_dir: Path) -> Path:
        """生成模拟 safetensors（可被服务端识别格式，但权重是假的）"""
        output_dir.mkdir(parents=True, exist_ok=True)
        lora_path = output_dir / "lora_weights.safetensors"

        rng = np.random.default_rng(42)
        tensors = {
            "lora_A.weight": rng.standard_normal((16, 64)).astype(np.float32),
            "lora_B.weight": rng.standard_normal((64, 16)).astype(np.float32),
        }
        st_save(tensors, str(lora_path))
        return lora_path
