"""QLoRA 微调训练模块"""
import os
import gc
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import boto3
from botocore.config import Config as BotoConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from firefly.core.client import TaskPackage
from firefly.core.hardware import get_hardware_info

logger = logging.getLogger("firefly.training")


@dataclass
class TrainingResult:
    final_loss: float | None
    steps_completed: int
    epoch_completed: float
    lora_weights_path: str | None
    log_summary: str | None


class S3Downloader:
    """从 MinIO / S3 下载训练数据"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )
        self.bucket = bucket

    def download_prefix(self, s3_prefix: str, local_dir: Path):
        """下载 s3://bucket/s3_prefix/ 下所有文件到本地目录"""
        local_dir.mkdir(parents=True, exist_ok=True)
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local_path = local_dir / os.path.relpath(key, s3_prefix)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, key, str(local_path))
                logger.info(f"Downloaded: {key} -> {local_path}")


class QLoRATrainer:
    """
    QLoRA 微调训练器。
    v0.1 支持：
    - 4-bit量化（QLoRA）
    - LoRA adapter
    - 断点续训
    """

    def __init__(
        self,
        scheduler_url: str,
        s3_endpoint: str,
        s3_access_key: str,
        s3_secret_key: str,
        s3_bucket: str,
        output_dir: str = "./firefly_output",
    ):
        self.scheduler_url = scheduler_url
        self.s3_bucket = s3_bucket
        self.downloader = S3Downloader(s3_endpoint, s3_access_key, s3_secret_key, s3_bucket)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None

    def _get_device_map(self) -> dict:
        """根据 GPU 数量生成 device_map"""
        hw = get_hardware_info()
        if hw.gpu_count >= 2 and torch.cuda.device_count() >= 2:
            # 多卡：自动分配
            return "auto"
        return {"": 0}

    async def train(self, task: TaskPackage) -> TrainingResult:
        """
        执行一个训练任务包。
        流程：下载数据 → 加载模型 → 训练 → 上传权重 → 清理
        """
        local_data_dir = self.output_dir / "data" / task.task_id
        logger.info(f"开始训练任务 {task.task_id}，基础模型: {task.base_model}")

        try:
            # Step 1: 下载训练数据
            self.downloader.download_prefix(task.train_data_s3_prefix, local_data_dir)

            # Step 2: 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                task.base_model,
                trust_remote_code=True,
                use_fast=False,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Step 3: 加载 4-bit 量化模型
            quant_cfg = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
            self._model = AutoModelForCausalLM.from_pretrained(
                task.base_model,
                quantization_config=quant_cfg,
                device_map=self._get_device_map(),
                trust_remote_code=True,
            )
            self._model = prepare_model_for_kbit_training(self._model)

            # Step 4: 加载数据集（JSONL 格式）
            jsonl_files = list(local_data_dir.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"未找到训练数据文件: {local_data_dir}/*.jsonl")

            dataset = load_dataset("json", data_files=str(jsonl_files[0]), split="train")

            def tokenize_fn(example):
                text = example.get("text", "")
                return self._tokenizer(
                    text,
                    truncation=True,
                    max_length=task.config.get("max_length", 2048),
                )

            dataset = dataset.map(tokenize_fn, remove_columns=["text"])

            # Step 5: 配置 LoRA
            lora_cfg = LoraConfig(
                r=task.config.get("lora_r", 16),
                lora_alpha=task.config.get("lora_alpha", 32),
                target_modules=task.config.get("lora_target_modules", ["q_proj", "v_proj"]),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, lora_cfg)
            self._model.print_trainable_parameters()

            # Step 6: 训练参数
            run_name = f"firefly-{task.task_id[:8]}"
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / run_name),
                num_train_epochs=task.config.get("num_epochs", 1),
                per_device_train_batch_size=task.config.get("per_device_train_batch_size", 2),
                gradient_accumulation_steps=task.config.get("gradient_accumulation_steps", 4),
                learning_rate=task.config.get("learning_rate", 2e-4),
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                logging_steps=task.config.get("logging_steps", 10),
                save_steps=task.config.get("save_steps", 100),
                save_total_limit=1,
                bf16=torch.cuda.is_bf16_supported(),
                fp16=not torch.cuda.is_bf16_supported(),
                report_to=["none"],
                run_name=run_name,
                gradient_checkpointing=True,
                max_grad_norm=0.3,
            )

            # Step 7: 开始训练
            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(self._tokenizer, mlm=False),
            )
            trainer.train()

            # Step 8: 保存 LoRA 权重到本地
            weights_dir = self.output_dir / "weights" / task.submission_id_key
            self._model.save_pretrained(str(weights_dir))

            final_loss = trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None
            steps = trainer.state.global_step
            epochs = trainer.state.num_train_epochs

            return TrainingResult(
                final_loss=final_loss,
                steps_completed=steps,
                epoch_completed=epochs,
                lora_weights_path=str(weights_dir),
                log_summary=f"Loss={final_loss:.4f}, Steps={steps}, Epochs={epochs:.1f}",
            )

        except Exception as e:
            logger.exception(f"训练失败: {e}")
            raise

        finally:
            # Step 9: 清理显存
            del self._model
            del self._tokenizer
            gc.collect()
            torch.cuda.empty_cache()
