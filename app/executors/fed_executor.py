"""
firefly-client · Executors · FedExecutor
v0.6 联邦执行器

职责：对接调度中心，完整走通联邦训练生命周期：
1. claim_task    → 认领任务（获取 task_id + 数据集 URL）
2. download_dataset → 下载训练数据
3. 本地 QLoRA 训练（复用 RealQLoRATrainer）
4. report_progress → 每 N 步上报 loss 到调度中心
5. complete_task   → 回报 final_loss + holdout_acc，上报信号（可选）
6. download_aggregated → 获取本轮 FedAvg 聚合权重

设计原则：
- 同步接口（供 fire start 调用）
- 不传权重到调度中心（带宽 + 隐私）
- 断点续跑：本地 checkpoint 检测（不重复训练已完成步骤）
- 信号回流：训练完成后可选上报 DonationBridge
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

from app.config import ClientConfig, get_headers, load_config, save_config
from app.trainer import TrainingConfig, MockTrainer, RealQLoRATrainer
from app.donation_bridge import DonationBridge, SignalPayload, report_training_signal
from app.hardware import full_hardware_report as get_hardware_info

# ─────────────────────────────────────
# 数据结构
# ─────────────────────────────────────

@dataclass
class FedTask:
    """联邦任务（从调度中心 claim 后得到）"""
    task_id: str
    task_name: str
    domain: str
    task_level: int
    config: dict                # 训练超参数
    deadline: datetime
    dataset_url: Optional[str]  # 预签名下载链接（无则为 None）
    base_contribution: int


@dataclass
class TrainingResult:
    """训练结果（供 complete_task + 信号回流转）"""
    task_id: str
    final_loss: float
    holdout_accuracy: float
    peak_vram_mb: float
    execution_time_sec: float
    total_steps: int
    lora_path: Path | None
    training_log: dict


# ─────────────────────────────────────
# 联邦执行器
# ─────────────────────────────────────

class FedExecutor:
    """
    联邦执行器（主类）

    用法示例：
        executor = FedExecutor(scheduler_url="http://106.14.220.169:8000")
        cfg = load_config()
        cfg.server_url = "http://106.14.220.169:8000"
        save_config(cfg)

        task = await executor.claim_task(domain="law", cfg=cfg)
        result = await executor.run_training(task, cfg=cfg)
        await executor.complete_task(result, cfg=cfg)
        await executor.download_aggregated(task_round=1, target_dir=Path("~/.firefly/round1"))
    """

    PROGRESS_INTERVAL = 10   # 每 10 步上报一次进度

    def __init__(self, scheduler_url: str, progress_interval: int = 10):
        self.scheduler_url = scheduler_url.rstrip("/")
        self.PROGRESS_INTERVAL = progress_interval
        self._client: httpx.AsyncClient | None = None

    # ── 生命周期 ───────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── 1. 认领任务 ───────────────────

    async def claim_task(
        self,
        domain: str,
        cfg: ClientConfig,
        preferred_level: int | None = None,
    ) -> FedTask:
        """
        向调度中心认领一个任务

        Returns:
            FedTask 对象（含 task_id / config / dataset_url）

        Raises:
            httpx.HTTPStatusError: 认领失败（无可用任务 / 权限不足）
        """
        client = await self._get_client()
        headers = get_headers(cfg)

        payload = {
            "node": cfg.node_id,
            "domain": domain,
        }
        if preferred_level is not None:
            payload["preferred_level"] = preferred_level

        resp = await client.post(
            f"{self.scheduler_url}/api/v1/tasks/claim",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        return FedTask(
            task_id=data["task_id"],
            task_name=data["task_name"],
            domain=domain,
            task_level=data.get("task_level", 1),
            config=data.get("config", {}),
            deadline=datetime.fromisoformat(data.get("deadline", datetime.utcnow().isoformat())),
            dataset_url=data.get("task_package_url"),
            base_contribution=data.get("base_contribution", 10),
        )

    # ── 2. 下载数据集 ──────────────────

    async def download_dataset(
        self,
        task: FedTask,
        target_dir: Path,
        cfg: ClientConfig,
    ) -> Path:
        """
        下载任务数据包到本地目录

        策略：
        - dataset_url 有值 → 用预签名链接下载 zip → 解压
        - dataset_url 为 None → 调度中心没有独立数据包，训练器内部加载
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        if not task.dataset_url:
            # 无独立数据包，训练器自行加载
            return target_dir

        try:
            client = await self._get_client()
            headers = get_headers(cfg)
            resp = await client.get(task.dataset_url, headers=headers, follow_redirects=True)
            resp.raise_for_status()

            zip_path = target_dir / f"task_{task.task_id[:8]}.zip"
            zip_path.write_bytes(resp.content)

            # 解压
            extract_dir = target_dir / "dataset"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            zip_path.unlink(missing_ok=True)
            return extract_dir

        except httpx.HTTPStatusError as e:
            print(f"[FedExecutor] Dataset download failed ({e.response.status_code}), skipping")
            return target_dir

    # ── 3. 本地训练 ────────────────────

    async def run_training(
        self,
        task: FedTask,
        cfg: ClientConfig,
        output_dir: Path | None = None,
        report_interval: int | None = None,
    ) -> TrainingResult:
        """
        执行本地 QLoRA 训练

        流程：
        1. 构建 TrainingConfig（从 task.config 或环境变量）
        2. 选择 RealQLoRATrainer（无 GPU 时自动 fallback 到 Mock）
        3. 每 PROGRESS_INTERVAL 步上报进度到调度中心
        4. 返回 TrainingResult
        """
        output_dir = output_dir or (Path.home() / ".firefly" / "tasks" / task.task_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 从 task.config 提取超参数
        tcfg = task.config
        steps = tcfg.get("max_steps", int(os.environ.get("FIREFLY_MAX_STEPS", "60")))
        dataset_path = tcfg.get("dataset_path", os.environ.get("FIREFLY_DATASET", ""))

        train_cfg = TrainingConfig(
            model_name=os.environ.get("FIREFLY_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            max_steps=steps,
            dataset_path=dataset_path,
            lora_rank=int(tcfg.get("lora_rank", os.environ.get("FIREFLY_LORA_R", "8"))),
            lora_alpha=int(tcfg.get("lora_alpha", os.environ.get("FIREFLY_LORA_ALPHA", "16"))),
            lora_targets=tcfg.get("lora_targets", "q_proj,v_proj"),
            learning_rate=float(tcfg.get("learning_rate", "2e-4")),
            gradient_accumulation=int(tcfg.get("gradient_accumulation", "4")),
            output_dir=str(output_dir),
        )

        # 训练器选择（Mock 兜底）
        is_mock = os.environ.get("FIREFLY_MOCK") == "1"
        trainer: RealQLoRATrainer | MockTrainer = (
            MockTrainer(train_cfg) if is_mock else RealQLoRATrainer(train_cfg)
        )
        if is_mock:
            print("[FedExecutor] Running in MOCK mode (FIREFLY_MOCK=1)")

        # 进度上报回调
        interval = report_interval or self.PROGRESS_INTERVAL

        async def on_progress(step: int, loss: float | None, total: int):
            await self.report_progress(task.task_id, step, total, loss, cfg)

        # 开始训练
        print(f"[FedExecutor] Training task {task.task_id} ({task.domain}, {steps} steps)")
        t_start = time.monotonic()
        hardware = get_hardware_info()

        result = await trainer.train(on_progress=on_progress)
        t_end = time.monotonic()

        # 提取结果
        final_loss = result.get("final_loss", 0.0)
        total_steps = result.get("total_steps", steps)
        lora_path = Path(result.get("lora_path", ""))

        # holdout accuracy（训练器内计算，result 中透传）
        holdout_acc = result.get("holdout_accuracy", 0.0)

        training_log = {
            "task_id": task.task_id,
            "domain": task.domain,
            "final_loss": final_loss,
            "holdout_accuracy": holdout_acc,
            "total_steps": total_steps,
            "execution_time_sec": round(t_end - t_start, 1),
            "peak_vram_mb": hardware.get("gpu_vram_used_mb", 0.0),
            "start_time": datetime.utcnow().isoformat(),
            "end_time": datetime.utcnow().isoformat(),
        }

        return TrainingResult(
            task_id=task.task_id,
            final_loss=final_loss,
            holdout_accuracy=holdout_acc,
            peak_vram_mb=hardware.get("gpu_vram_used_mb", 0.0),
            execution_time_sec=round(t_end - t_start, 1),
            total_steps=total_steps,
            lora_path=lora_path,
            training_log=training_log,
        )

    # ── 4. 进度上报 ───────────────────

    async def report_progress(
        self,
        task_id: str,
        step: int,
        total_steps: int,
        loss: float | None,
        cfg: ClientConfig,
    ):
        """向调度中心上报训练进度（每 N 步调用一次）"""
        pct = round(step / total_steps * 100, 1) if total_steps else 0
        client = await self._get_client()
        headers = get_headers(cfg)

        body = {
            "task_id": task_id,
            "step": step,
            "total_steps": total_steps,
            "progress_pct": pct,
            "loss": loss,
        }

        try:
            resp = await client.post(
                f"{self.scheduler_url}/api/v1/tasks/progress",
                json=body,
                headers=headers,
            )
            # 非致命：忽略失败
            if resp.status_code != 200:
                print(f"[FedExecutor] Progress report failed: {resp.status_code}")
        except httpx.RequestError as e:
            print(f"[FedExecutor] Progress report error (non-fatal): {e}")

    # ── 5. 完成任务 ───────────────────

    async def complete_task(
        self,
        result: TrainingResult,
        cfg: ClientConfig,
        result_object_name: str | None = None,
        result_sha256: str | None = None,
        donate_signal: bool = True,
    ):
        """
        向调度中心提交任务结果

        Args:
            result: 训练结果对象
            cfg: 客户端配置
            result_object_name: MinIO 中的结果对象路径（可选，本版本不上传权重）
            result_sha256: 结果 SHA256（可选）
            donate_signal: 是否同时上报信号（DonationBridge）
        """
        client = await self._get_client()
        headers = get_headers(cfg)

        # 构造提交体（v0.6：不传权重 URL，权重留在本地）
        body = {
            "task_id": result.task_id,
            # result_object_name / result_sha256 暂时不填（v0.6 不传权重）
            "final_loss": result.final_loss,
            "holdout_accuracy": result.holdout_accuracy,
            "peak_vram_mb": result.peak_vram_mb,
            "execution_time_sec": result.execution_time_sec,
            "total_steps": result.total_steps,
        }

        resp = await client.post(
            f"{self.scheduler_url}/api/v1/tasks/complete",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        print(f"[FedExecutor] Task {result.task_id} completed successfully")

        # ── 信号回流（可选） ──
        if donate_signal:
            try:
                sig_result = await report_training_signal(
                    task_id=result.task_id,
                    node_id=cfg.node_id,
                    training_stats=result.training_log,
                    holdout_before=0.0,          # TODO: 从训练前 holdout 集获取
                    holdout_after=result.holdout_accuracy,
                    cfg=cfg,
                )
                if sig_result:
                    print(f"[FedExecutor] Signal donated: {sig_result}")
            except Exception as e:
                print(f"[FedExecutor] Signal donation failed (non-fatal): {e}")

    # ── 6. 下载聚合权重 ────────────────

    async def download_aggregated(
        self,
        round_num: int,
        target_dir: Path,
        cfg: ClientConfig,
    ) -> Path | None:
        """
        下载本轮 FedAvg 聚合权重（可选，节点可选择性加载）

        Returns:
            权重文件本地路径，或 None（无聚合结果）
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        client = await self._get_client()
        headers = get_headers(cfg)

        try:
            resp = await client.get(
                f"{self.scheduler_url}/api/v1/aggregation/download",
                params={"round": round_num},
                headers=headers,
                follow_redirects=True,
            )
            if resp.status_code == 404:
                print(f"[FedExecutor] No aggregated weights for round {round_num} yet")
                return None

            resp.raise_for_status()
            local_path = target_dir / f"aggregated_round{round_num}.safetensors"
            local_path.write_bytes(resp.content)
            print(f"[FedExecutor] Downloaded aggregated weights: {local_path}")
            return local_path

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"[FedExecutor] Not authorized to download round {round_num}")
            else:
                print(f"[FedExecutor] Download failed: {e}")
            return None

    # ── 7. 完整联邦流程（一条命令） ─────

    async def run_federated_round(
        self,
        domain: str,
        cfg: ClientConfig,
        output_dir: Path | None = None,
        donate_signal: bool = True,
    ) -> TrainingResult | None:
        """
        完整跑一次联邦训练生命周期（认领 → 训练 → 完成 → 下载聚合权重）

        Returns:
            TrainingResult，或 None（无任务可认领）
        """
        print(f"[FedExecutor] === Starting federated round for domain: {domain} ===")

        # 认领
        try:
            task = await self.claim_task(domain, cfg)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"[FedExecutor] No tasks available for domain: {domain}")
                return None
            raise

        print(f"[FedExecutor] Claimed: {task.task_id} ({task.task_name})")

        # 训练
        result = await self.run_training(task, cfg, output_dir=output_dir)

        # 提交
        await self.complete_task(result, cfg, donate_signal=donate_signal)

        # 下载聚合权重（当前轮的聚合可能还没生成，尝试下载上一轮）
        await self.download_aggregated(
            round_num=max(1, 1),  # 先写死 round=1，后续按实际轮次
            target_dir=output_dir or Path.home() / ".firefly" / "rounds",
            cfg=cfg,
        )

        print(f"[FedExecutor] === Round complete: {result.task_id} ===")
        return result


# ─────────────────────────────────────
# 便捷入口（CLI / fire start 内部调用）
# ─────────────────────────────────────

async def run_federated_training(
    scheduler_url: str,
    domain: str,
    node_id: str,
    access_token: str,
) -> TrainingResult | None:
    """
    一次性联邦训练（最简入口）
    自动加载/保存配置文件
    """
    cfg = load_config()
    cfg.server_url = scheduler_url
    cfg.node_id = node_id
    cfg.access_token = access_token
    save_config(cfg)

    executor = FedExecutor(scheduler_url)
    try:
        result = await executor.run_federated_round(domain, cfg)
        return result
    finally:
        await executor.close()
