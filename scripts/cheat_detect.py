# -*- coding: utf-8 -*-
"""
防作弊 L1：cos 相似度检测
聚合前对比各 adapter 与基线（第一个 adapter）的 cos 相似度
低于阈值的 adapter 标记为可疑，权重清零

用法：
  python scripts/cheat_detect.py \
    --adapters outputs/law_r5_node1/adapter.safetensors,outputs/law_r5_node2/adapter.safetensors \
    --threshold 0.10

输出：
  - 终端打印每个 adapter 的相似度分数
  - 退出码 0 = 全部通过，1 = 有可疑 adapter
  - 可被 fedavg_weighted.py 调用做聚合前检查
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import torch
    from safetensors.torch import load_file
except Exception as e:
    print(f"[cheat_detect] 缺少 safetensors/torch: {e}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="防作弊 L1：cos 相似度检测")
    parser.add_argument("--adapters", type=str, required=True,
                        help="逗号分隔的 adapter 路径列表")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="cos 相似度阈值，低于此值标记为可疑（默认 0.10）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出路径（JSON），不指定则只打印")
    return parser.parse_args()


def cos_similarity(tensor_a, tensor_b):
    """计算两个张量的余弦相似度"""
    a = tensor_a.flatten().float()
    b = tensor_b.flatten().float()
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def compute_pairwise_similarity(adapters):
    """
    计算所有 adapter 与第一个 adapter 的 cos 相似度
    返回 [{index, path, similarity, keys_matched}] 列表
    """
    results = []
    base_tensors = load_file(adapters[0])
    base_keys = set(base_tensors.keys())

    for i, path in enumerate(adapters):
        if i == 0:
            results.append({
                "index": 0,
                "path": path,
                "similarity": 1.0,
                "keys_matched": len(base_keys),
                "status": "baseline",
            })
            continue

        tensors = load_file(path)
        keys = set(tensors.keys())

        # 检查键是否匹配
        if keys != base_keys:
            print(f"  [{i}] 键不匹配: 差异={len(keys ^ base_keys)} 个")
            results.append({
                "index": i,
                "path": path,
                "similarity": 0.0,
                "keys_matched": len(keys & base_keys),
                "status": "key_mismatch",
            })
            continue

        # 计算所有共享 key 的平均 cos 相似度
        sims = []
        for key in sorted(base_keys & keys):
            sim = cos_similarity(base_tensors[key], tensors[key])
            sims.append(sim)

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        results.append({
            "index": i,
            "path": path,
            "similarity": avg_sim,
            "keys_matched": len(keys & base_keys),
            "status": "ok" if avg_sim >= 0 else "negative",
        })

    return results


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

    print(f"防作弊检测: {len(adapters)} 个 adapter")
    print(f"阈值: {args.threshold}")
    print()

    results = compute_pairwise_similarity(adapters)

    suspicious = []
    print(f"{'索引':>4} {'相似度':>10} {'状态':>12} {'路径'}")
    print("-" * 80)
    for r in results:
        name = os.path.basename(os.path.dirname(r["path"]))
        sim_str = f"{r['similarity']:.4f}" if r["similarity"] >= 0 else "N/A"
        print(f"{r['index']:>4} {sim_str:>10} {r['status']:>12} {name}")

        if r["status"] not in ("baseline", "ok") or r["similarity"] < args.threshold:
            suspicious.append(r)

    print()
    if suspicious:
        print(f"[!] {len(suspicious)} 个可疑 adapter (相似度 < {args.threshold}):")
        for r in suspicious:
            name = os.path.basename(os.path.dirname(r["path"]))
            print(f"  - [{r['index']}] {name}: sim={r['similarity']:.4f}, status={r['status']}")
        print("\n建议: 可疑 adapter 权重清零，不参与聚合")

        if args.output:
            report = {
                "threshold": args.threshold,
                "total_adapters": len(adapters),
                "suspicious_count": len(suspicious),
                "results": results,
                "recommendation": "zero_weight_suspicious",
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"报告: {args.output}")

        sys.exit(1)
    else:
        print(f"[OK] 全部 {len(adapters)} 个 adapter 通过检测")
        if args.output:
            report = {
                "threshold": args.threshold,
                "total_adapters": len(adapters),
                "suspicious_count": 0,
                "results": results,
                "recommendation": "all_clear",
            }
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        sys.exit(0)


if __name__ == "__main__":
    main()
