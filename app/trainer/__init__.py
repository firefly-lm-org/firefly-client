"""
firefly-client · 训练器模块
"""
from app.trainer.base import BaseTrainer, TrainingConfig
from app.trainer.mock_trainer import MockTrainer          # v0.1
from app.trainer.real_trainer import RealQLoRATrainer       # v0.2+

__all__ = [
    "BaseTrainer",
    "TrainingConfig",
    "MockTrainer",
    "RealQLoRATrainer",
]
