# -*- coding: utf-8 -*-
"""
FedAvg 软加权聚合（基于 holdout_improvement + sample_count）
替代等权平均，让贡献大的节点权重更大

用法：
  python scripts/fedavg_weighted.py \
    --adapters outputs/law_r5_node1/adapter.safetensors,outputs/law_r5_node2/adapter.safetensors \
    --signals signals.jsonl \
    --output outputs/aggregated/adapter_model.safetensors

权重计算规则：
  基础权重 = sample_count（样本越多权重越大）
  提升系数 = 1 + max(0, holdout_improvement)（效果越好权重越大）
  最终权重 = 基础权重 * 提升系数，归一化后裁剪到 [min_weight, 1-min_weight]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

try:
    import torch
    from safetensors.torch import load_file, save_file
except Exception as e:
    print(f"[fedavg_weighted] 缺少 safetensors/torch: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg 软加权聚合")
    parser.add_argument("--adapters", type=str, required=True,
                        help="逗号分隔的 adapter 路径列表")
    parser.add_argument("--signals", type=str, default=None,
                        help="信号文件路径（JSONL），用于计算权重")
    parser.add_argument("--output", type=str, required=True,
                        help="输出聚合权重路径")
    parser.add_argument("--min-weight", type=float, default=0.05,
                        help="最小权重阈值，防止节点被完全忽略")
    return parser.parse_args()


def load_signals(signals_path):
    """加载信号文件，返回 {task_id: signal_dict}"""
    if not signals_path or not os.path.exists(signals_path):
        return {}

    signals = {}
    with open(signals_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sig = json.loads(line)
                task_id = sig.get("task_id", "")
                signals[task_id] = sig
    return signals


def compute_weights(adapters, signals, min_weight=0.05):
    """
    计算每个 adapter 的聚合权重

    规则：
    - 基础权重 = sample_count（样本越多权重越大）
    - 提升系数 = 1 + max(0, holdout_improvement)（效果越好权重越大）
    - 最终权重 = 基础权重 * 提升系数
    - 归一化后裁剪到 [min_weight, 1-min_weight]
    """
    n = len(adapters)
    raw_weights = []

    for i, adapter_path in enumerate(adapters):
        # 从路径推断 task_id（取倒数第二个目录名）
        parts = adapter_path.replace("\\", "/").split("/")
        task_candidate = parts[-2] if len(parts) >= 2 else f"node_{i}"

        # 查找对应信号
        signal = None
        for tid, sig in signals.items():
            if tid.startswith(task_candidate) or task_candidate in tid:
                signal = sig
                break

        if signal:
            sample_count = signal.get("sample_count", 10)
            holdout_imp = signal.get("holdout_improvement", 0)
            improvement_factor = 1.0 + max(0, holdout_imp)
            weight = sample_count * improvement_factor
            print(f"  [{i}] {task_candidate}: samples={sample_count}, "
                  f"improvement={holdout_imp:.4f}, raw_weight={weight:.2f}")
        else:
            # 没有信号时用等权
            weight = 1.0
            print(f"  [{i}] {task_candidate}: 无信号，使用等权")

        raw_weights.append(weight)

    # 归一化
    total = sum(raw_weights)
    if total == 0:
        weights = [1.0 / n] * n
    else:
        weights = [w / total for w in raw_weights]

    # 裁剪
    for i in range(n):
        if weights[i] < min_weight:
            weights[i] = min_weight
        elif weights[i] > 1.0 - min_weight:
            weights[i] = 1.0 - min_weight

    # 重新归一化
    total = sum(weights)
    weights = [w / total for w in weights]

    return weights


def weighted_aggregation(adapters, weights, output_path):
    """加权聚合所有 adapter"""
    print(f"\n聚合权重:")
    for i, (path, w) in enumerate(zip(adapters, weights)):
        print(f"  [{i}] {os.path.basename(os.path.dirname(path))}: weight={w:.4f}")

    # 加载所有权重
    all_tensors = []
    keys0 = None
    for path in adapters:
        t = load_file(path)
        if keys0 is None:
            keys0 = set(t.keys())
        elif set(t.keys()) != keys0:
            print(f"[fedavg_weighted] 张量键不匹配 {path}")
            sys.exit(1)
        all_tensors.append(t)

    # 加权平均
    aggregated = {}
    for key in sorted(keys0):
        stacked = torch.stack([t[key].float() for t in all_tensors])
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        while weight_tensor.dim() < stacked.dim():
            weight_tensor = weight_tensor.unsqueeze(-1)
        aggregated[key] = (stacked * weight_tensor).sum(dim=0)

    # 保存
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    save_file(aggregated, output_path)

    # 计算 SHA256
    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    print(f"\n输出: {output_path}")
    print(f"SHA256: {sha.hexdigest()[:16]}...")
    print(f"Tensor 数: {len(aggregated)}")

    return sha.hexdigest()


def main():
    args = parse_args()

    adapters = [p.strip() for p in args.adapters.split(",")]
    if len(adapters) < 2:
        print("错误: 至少需要 2 个 adapter")
        sys.exit(1)

    for path in adapters:
        if not os.path.exists(path):
            print(f"错误: adapter 不存在: {path}")
            sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 加载信号
    signals = load_signals(args.signals)
    print(f"加载信号: {len(signals)} 条")

    # 计算权重
    weights = compute_weights(adapters, signals, args.min_weight)

    # 聚合
    sha = weighted_aggregation(adapters, weights, args.output)

    # 输出权重信息供下游使用
    weight_info = {
        "method": "weighted_fedavg",
        "adapters": adapters,
        "weights": weights,
        "sha256": sha,
        "num_tensors": 0,
    }
    weight_info_path = args.output.replace(".safetensors", "_weights.json")
    with open(weight_info_path, "w", encoding="utf-8") as f:
        json.dump(weight_info, f, indent=2, ensure_ascii=False)
    print(f"权重信息: {weight_info_path}")


if __name__ == "__main__":
    main()
