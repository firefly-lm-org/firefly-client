"""
firefly-client · Donation Bridge
v0.6 信号回流协议

节点训练完成后，可选贡献信号（不传权重）：
1. holdout_acc_improvement：相对基准的提升量（标量）
2. sample_count：愿意授权的脱敏改写样本数
3. query_patterns：高频 query 统计特征（去标识化）

设计原则：
- 只传信号，不传权重（隐私 + 带宽）
- 贡献完全自愿（opt-in）
- 服务器端聚合统计特征，用于调度优化
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from pathlib import Path
from typing import Optional

import httpx

from app.config import ClientConfig, get_headers

__all__ = ["DonationBridge", "SignalPayload"]


# ─────────────────────────────────────
# 数据结构
# ─────────────────────────────────────

class SignalPayload:
    """
    信号载体（客户端构造，服务端解析）

    设计要点：
    - query_patterns 已在客户端做 min-hash 去标识化
    - holdout_improvement 是 delta，不含原始分数
    - sample_count 仅计数，不传内容（服务端根据计数奖励）
    """

    def __init__(
        self,
        task_id: str,
        node_id: str,
        holdout_improvement: float = 0.0,
        baseline_accuracy: float = 0.0,
        final_accuracy: float = 0.0,
        sample_count: int = 0,
        query_patterns: list[str] | None = None,
        total_steps: int = 0,
        final_loss: float = 0.0,
        training_time_sec: float = 0.0,
        gpu_model: str = "",
        gpu_vram_gb: float = 0.0,
    ):
        self.task_id = task_id
        self.node_id = node_id
        self.holdout_improvement = holdout_improvement
        self.baseline_accuracy = baseline_accuracy
        self.final_accuracy = final_accuracy
        self.sample_count = sample_count
        # query_patterns：min-hash fingerprint list，每条是 8 字符 hex
        self.query_patterns = query_patterns or []
        self.total_steps = total_steps
        self.final_loss = final_loss
        self.training_time_sec = training_time_sec
        self.gpu_model = gpu_model
        self.gpu_vram_gb = gpu_vram_gb

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "holdout_improvement": round(self.holdout_improvement, 6),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "final_accuracy": round(self.final_accuracy, 4),
            "sample_count": self.sample_count,
            "query_patterns": self.query_patterns,
            "total_steps": self.total_steps,
            "final_loss": round(self.final_loss, 6),
            "training_time_sec": round(self.training_time_sec, 1),
            "gpu_model": self.gpu_model,
            "gpu_vram_gb": round(self.gpu_vram_gb, 1),
            "os_type": platform.system(),
            "client_version": "0.6",
            "idempotency_key": self._idempotency_key(),
        }

    def _idempotency_key(self) -> str:
        """防止重复上报的幂等键"""
        raw = f"{self.task_id}:{self.node_id}:{self.final_loss}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ─────────────────────────────────────
# 贡献桥
# ─────────────────────────────────────

class DonationBridge:
    """
    信号回流通道

    用法（训练完成后）：
        bridge = DonationBridge(cfg)
        signal = SignalPayload(
            task_id="xxx",
            node_id="yyy",
            holdout_improvement=0.12,   # 相对基准提升 12%
            sample_count=5,
            query_patterns=["a3f9b2c1", ...],
            ...
        )
        bridge.report_signal(signal)
    """

    def __init__(self, config: ClientConfig | None = None):
        self.cfg = config or ClientConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.cfg.scheduler_url}/api/v1/contrib/signal"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        return self._client

    async def report_signal(self, signal: SignalPayload) -> dict | None:
        """
        上报脱敏信号到调度中心

        Returns:
            服务端返回的贡献凭证，或 None（网络失败，不阻塞主流程）
        """
        payload = signal.to_dict()
        headers = {
            **get_headers(self.cfg),
            "Content-Type": "application/json",
            "X-Idempotency-Key": signal._idempotency_key(),
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                self.endpoint,
                json=payload,
                headers=headers,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 409:
                # 幂等：已上报过，跳过
                return {"status": "duplicate", "task_id": signal.task_id}
            else:
                print(f"[DonationBridge] Server returned {resp.status_code}: {resp.text[:100]}")
                return None
        except httpx.RequestError as e:
            print(f"[DonationBridge] Network error (non-fatal): {e}")
            return None

    async def report_signal_batch(self, signals: list[SignalPayload]) -> list[dict | None]:
        """批量上报（单次 HTTP 连接）"""
        results = []
        for s in signals:
            results.append(await self.report_signal(s))
        return results

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ─────────────────────────────────────
# 辅助：构造 query_patterns（min-hash fingerprint）
# ─────────────────────────────────────

def extract_query_fingerprints(
    texts: list[str],
    n_fingerprints: int = 8,
) -> list[str]:
    """
    对文本列表提取 min-hash 指纹（极度简化版，v0.6 先跑通）
    实际生产应使用 datasketch.minhash.MinHash

    返回：n_fingerprints 个 8 字符 hex fingerprint
    """
    import hashlib

    fingerprints = []
    for i in range(n_fingerprints):
        # 对每条文本的 n-gram hash 取 min
        min_hash = float("inf")
        for text in texts:
            # 取 3-gram
            text_clean = text.lower().strip()
            ngrams = [text_clean[j:j+3] for j in range(max(0, len(text_clean)-2))]
            if not ngrams:
                ngrams = [text_clean]
            for ng in ngrams:
                raw = f"{i}:{ng}".encode()
                h = int(hashlib.md5(raw).hexdigest(), 16)
                if h < min_hash:
                    min_hash = h
        fingerprints.append(f"{min_hash % (16**8):08x}")
    return fingerprints


# ─────────────────────────────────────
# 端到端辅助：训练完成后一键上报
# ─────────────────────────────────────

async def report_training_signal(
    task_id: str,
    node_id: str,
    training_stats: dict,
    holdout_before: float = 0.0,
    holdout_after: float = 0.0,
    donated_samples: list[str] | None = None,
    cfg: ClientConfig | None = None,
) -> dict | None:
    """
    训练完成后，一行调用上报信号

    Args:
        task_id: 任务 ID
        node_id: 节点 ID
        training_stats: 训练统计 dict（final_loss / total_steps / training_time_sec）
        holdout_before: 训练前 holdout 准确率
        holdout_after: 训练后 holdout 准确率
        donated_samples: 愿意贡献的脱敏样本文本列表（opt-in）
        cfg: ClientConfig（None 时从环境变量构造）
    """
    improvement = holdout_after - holdout_before
    patterns = []
    if donated_samples:
        patterns = extract_query_fingerprints(donated_samples)

    signal = SignalPayload(
        task_id=task_id,
        node_id=node_id,
        holdout_improvement=improvement,
        baseline_accuracy=holdout_before,
        final_accuracy=holdout_after,
        sample_count=len(donated_samples) if donated_samples else 0,
        query_patterns=patterns,
        total_steps=training_stats.get("total_steps", 0),
        final_loss=training_stats.get("final_loss", 0.0),
        training_time_sec=training_stats.get("training_time_sec", 0.0),
        gpu_model=training_stats.get("gpu_model", ""),
        gpu_vram_gb=training_stats.get("gpu_vram_gb", 0.0),
    )

    bridge = DonationBridge(cfg)
    try:
        result = await bridge.report_signal(signal)
        return result
    finally:
        await bridge.close()
