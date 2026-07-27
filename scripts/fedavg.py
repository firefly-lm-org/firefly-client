# -*- coding: utf-8 -*-
"""
FedAvg 聚合：对 N 个 LoRA safetensors 做（加权）算术平均。

前提：所有输入权重来自同一基座模型 + 相同 target_modules + 相同 rank，
      否则张量键不匹配会直接报错（这是 FedAvg 正确性的必要前提）。

用法：
  # 等权聚合（第一轮推荐）
  python scripts/fedavg.py w1.safetensors w2.safetensors w3.safetensors w4.safetensors \
      --output ./aggregated_round1/adapter_model.safetensors

  # 按样本数加权（系数自动归一化；例如 law10/med31/py30/tax29 -> 0.10/0.31/0.30/0.29）
  python scripts/fedavg.py w1.safetensors w2.safetensors w3.safetensors w4.safetensors \
      --weights-coef 0.10 0.31 0.30 0.29 \
      --output ./aggregated_round1/adapter_model.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from safetensors.torch import load_file, save_file
except Exception as e:
    print(f"[fedavg] 缺少 safetensors/torch: {e}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Firefly FedAvg 聚合")
    ap.add_argument("weights", nargs="+", help="参与聚合的 safetensors 路径（>=2）")
    ap.add_argument("--weights-coef", nargs="+", type=float, default=None,
                    help="加权系数（样本数/损失等）；缺省等权，自动归一化")
    ap.add_argument("--output", required=True, help="聚合后 safetensors 输出路径")
    args = ap.parse_args()

    paths = args.weights
    if len(paths) < 2:
        print("[fedavg] 至少需要 2 个权重")
        sys.exit(1)

    coefs = args.weights_coef
    if coefs is None:
        coefs = [1.0 / len(paths)] * len(paths)
    else:
        if len(coefs) != len(paths):
            print("[fedavg] --weights-coef 数量必须与权重数量一致")
            sys.exit(1)
        s = sum(coefs)
        coefs = [c / s for c in coefs]  # 归一化

    tensors = []
    keys0 = None
    for p in paths:
        if not os.path.exists(p):
            print(f"[fedavg] 文件缺失: {p}")
            sys.exit(1)
        t = load_file(p)
        if keys0 is None:
            keys0 = set(t.keys())
        elif set(t.keys()) != keys0:
            print(f"[fedavg] 张量键不匹配 {p}: 差异={set(t.keys()) ^ keys0}")
            sys.exit(1)
        tensors.append(t)

    print(f"[fedavg] 聚合 {len(paths)} 个权重，{len(keys0)} 个张量，"
          f"系数={[round(c, 3) for c in coefs]}")

    merged = {}
    for k in sorted(keys0):
        acc = None
        for t, c in zip(tensors, coefs):
            v = t[k].float() * c
            acc = v if acc is None else acc + v
        merged[k] = acc

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    save_file(merged, args.output)

    meta = {
        "method": "fedavg",
        "n": len(paths),
        "keys": len(keys0),
        "coefs": coefs,
        "source": paths,
    }
    with open(os.path.join(out_dir, "fedavg_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size = os.path.getsize(args.output)
    print(f"[fedavg] 已写出: {args.output} ({size} bytes)")
    print(f"[fedavg] meta: {os.path.join(out_dir, 'fedavg_meta.json')}")


if __name__ == "__main__":
    main()
