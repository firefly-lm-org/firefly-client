# -*- coding: utf-8 -*-
"""
Demo Mode: single-machine federated training simulation.
One command runs the full pipeline: data split -> mock train -> anti-cheat -> FedAvg -> mock eval.

Usage:
  python scripts/demo_mode.py
  python scripts/demo_mode.py --domain law --nodes 2 --steps 5
  firefly-node demo

Requirements:
  pip install torch safetensors

No GPU needed. No unsloth needed. No real model loading.
The mock training creates small random LoRA tensors so the full FLOW is real,
only the training and evaluation steps are simulated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for torch + safetensors
try:
    import torch
    from safetensors.torch import save_file, load_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Mock training ────────────────────────────

def create_mock_adapter(node_id: int, output_path: str, num_items: int, steps: int = 5) -> dict:
    """
    Create a mock LoRA adapter (small random tensors).
    Each node uses a different seed so adapters are genuinely different.
    Same keys + same shapes across nodes so cheat_detect and FedAvg can work.
    """
    torch.manual_seed(node_id * 42 + steps)

    rank = 8
    dim = 256  # small for speed; real Qwen3-1.5B uses 1536
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, dim),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(dim, rank),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, dim),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(dim, rank),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(tensors, output_path)

    # Simulate decreasing loss
    base_loss = 2.5
    final_loss = max(base_loss - steps * 0.05 - node_id * 0.01, 0.1)

    # SHA256
    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return {
        "node_id": node_id,
        "adapter_path": output_path,
        "final_loss": final_loss,
        "sha256": sha.hexdigest(),
        "sample_count": num_items,
        "holdout_improvement": 0.01 + node_id * 0.003,
    }


# ── Inline anti-cheat (same algorithm as cheat_detect.py) ────────────

def cheat_detect(adapters: list, threshold: float = 0.10) -> list:
    """
    Cos similarity detection (inlined to avoid cheat_detect.py sys.exit on import).
    Compares each adapter against the first (baseline).
    """
    base_tensors = load_file(adapters[0])
    base_keys = set(base_tensors.keys())

    results = []
    for i, path in enumerate(adapters):
        if i == 0:
            results.append({"index": 0, "path": path, "similarity": 1.0, "status": "baseline"})
            continue

        tensors = load_file(path)
        keys = set(tensors.keys())

        if keys != base_keys:
            results.append({"index": i, "path": path, "similarity": 0.0, "status": "key_mismatch"})
            continue

        sims = []
        for key in sorted(base_keys & keys):
            a = base_tensors[key].flatten().float()
            b = tensors[key].flatten().float()
            na = torch.norm(a)
            nb = torch.norm(b)
            if na > 0 and nb > 0:
                sims.append((torch.dot(a, b) / (na * nb)).item())

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        results.append({
            "index": i,
            "path": path,
            "similarity": avg_sim,
            "status": "ok" if avg_sim >= 0 else "negative",
        })

    return results


# ── Inline FedAvg (same algorithm as fedavg_weighted.py) ──────────────

def fedavg_weighted(adapters: list, signals: list, output_path: str, min_weight: float = 0.05) -> str:
    """
    Weighted FedAvg aggregation (inlined to avoid path-based task_id matching issues).
    weight = sample_count * (1 + relu(holdout_improvement)), normalized, clipped to [min_w, 1-min_w].
    """
    n = len(adapters)
    raw_weights = []
    for i, sig in enumerate(signals):
        sc = sig.get("sample_count", 10)
        hi = sig.get("holdout_improvement", 0)
        w = sc * (1.0 + max(0, hi))
        raw_weights.append(w)
        print(f"  [Node {i}] samples={sc}, improvement={hi:.4f}, raw_weight={w:.2f}")

    total = sum(raw_weights)
    weights = [w / total for w in raw_weights] if total > 0 else [1.0 / n] * n

    for i in range(n):
        if weights[i] < min_weight:
            weights[i] = min_weight
        elif weights[i] > 1.0 - min_weight:
            weights[i] = 1.0 - min_weight
    total = sum(weights)
    weights = [w / total for w in weights]

    print(f"\n  归一化权重:")
    for i, (path, w) in enumerate(zip(adapters, weights)):
        print(f"    [Node {i}] weight={w:.4f}")

    all_tensors = [load_file(p) for p in adapters]
    keys0 = set(all_tensors[0].keys())
    for t in all_tensors[1:]:
        if set(t.keys()) != keys0:
            print("  [ERROR] tensor keys mismatch")
            sys.exit(1)

    aggregated = {}
    for key in sorted(keys0):
        stacked = torch.stack([t[key].float() for t in all_tensors])
        wt = torch.tensor(weights, dtype=stacked.dtype)
        while wt.dim() < stacked.dim():
            wt = wt.unsqueeze(-1)
        aggregated[key] = (stacked * wt).sum(dim=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(aggregated, output_path)

    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    print(f"\n  输出: {output_path}")
    print(f"  SHA256: {sha.hexdigest()[:16]}...")
    print(f"  Tensor 数: {len(aggregated)}")
    return sha.hexdigest()


# ── Tolerant JSONL loader (handles unescaped quotes) ────────────────

def load_jsonl_tolerant(path):
    """
    Load JSONL with fallback regex parsing for lines with unescaped quotes.
    law_qa.jsonl line 16 has unescaped Chinese quotes inside JSON strings.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                instr = re.search(r'"instruction"\s*:\s*"(.+?)"\s*,\s*"output"', line)
                output = re.search(r'"output"\s*:\s*"(.+)"\s*}?\s*$', line)
                if instr and output:
                    items.append({
                        "instruction": instr.group(1),
                        "output": output.group(1).rstrip('"').rstrip("}"),
                    })
                else:
                    items.append({"instruction": line[:50], "output": ""})
    return items


# ── Main demo ─────────────────────────────────

def run_demo(domain: str = "law", num_nodes: int = 4, steps: int = 5, output_dir: str = "./demo_output") -> bool:
    """Run the full demo pipeline."""

    print("=" * 60)
    print("  Firefly LM - Demo Mode")
    print("  单机多节点联邦训练模拟")
    print("=" * 60)
    print(f"  领域: {domain}")
    print(f"  节点数: {num_nodes}")
    print(f"  训练步数: {steps}")
    print(f"  输出目录: {output_dir}")
    print()

    if not HAS_TORCH:
        print("[ERROR] 需要 torch + safetensors")
        print("  pip install torch safetensors")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Prepare data ──
    print("[1/7] 准备数据...")
    data_path = PROJECT_ROOT / "data" / f"{domain}_qa.jsonl"
    if not data_path.exists():
        data_path = Path(f"data/{domain}_qa.jsonl")
    if not data_path.exists():
        print(f"  [ERROR] 数据文件不存在: {data_path}")
        return False

    all_items = load_jsonl_tolerant(str(data_path))

    print(f"  加载 {len(all_items)} 条数据: {data_path}")

    items_per_node = max(1, len(all_items) // num_nodes)
    node_data_paths = []
    for i in range(num_nodes):
        start = i * items_per_node
        end = start + items_per_node if i < num_nodes - 1 else len(all_items)
        subset = all_items[start:end]
        node_path = os.path.join(output_dir, f"node_{i}_data.jsonl")
        with open(node_path, "w", encoding="utf-8") as f:
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        node_data_paths.append(node_path)
        print(f"  Node {i}: {len(subset)} 条 -> {node_path}")

    # ── Step 2: Mock training ──
    print(f"\n[2/7] 模拟训练 (mock LoRA adapters, {steps} steps)...")
    start_time = time.time()

    node_results = []
    for i in range(num_nodes):
        adapter_path = os.path.join(output_dir, f"node_{i}_adapter.safetensors")
        result = create_mock_adapter(i, adapter_path, items_per_node, steps)
        node_results.append(result)
        print(f"  Node {i}: loss={result['final_loss']:.4f}, sha={result['sha256'][:16]}...")

    train_time = time.time() - start_time
    print(f"  训练耗时: {train_time:.1f}s")

    # ── Step 3: Simulate claim/complete ──
    print("\n[3/7] 模拟联邦流程 (claim -> train -> complete)...")
    for nr in node_results:
        print(f"  Node {nr['node_id']}: claim -> train -> complete OK")

    # ── Step 4: Anti-cheat detection ──
    print("\n[4/7] 防作弊检测 (cos 相似度)...")
    adapter_paths = [nr["adapter_path"] for nr in node_results]
    cheat_results = cheat_detect(adapter_paths, threshold=0.10)

    print(f"  {'索引':>4} {'相似度':>10} {'状态':>12}")
    print("  " + "-" * 40)
    for r in cheat_results:
        sim_str = f"{r['similarity']:.4f}" if r["similarity"] >= 0 else "N/A"
        print(f"  {r['index']:>4} {sim_str:>10} {r['status']:>12}")

    suspicious = [r for r in cheat_results if r["status"] not in ("baseline", "ok") or r["similarity"] < 0.10]
    if suspicious:
        print(f"\n  [!] {len(suspicious)} 个可疑 adapter")
        print("  建议: 可疑 adapter 权重清零")
    else:
        print(f"\n  [OK] 全部 {len(adapter_paths)} 个 adapter 通过检测")

    # ── Step 5: FedAvg weighted aggregation ──
    print("\n[5/7] FedAvg 加权聚合...")

    signals = []
    for nr in node_results:
        signals.append({
            "task_id": f"demo_{domain}_node_{nr['node_id']}",
            "node_id": f"demo_node_{nr['node_id']}",
            "holdout_improvement": nr["holdout_improvement"],
            "sample_count": nr["sample_count"],
            "final_loss": nr["final_loss"],
        })

    # Save signals for reproducibility
    signals_path = os.path.join(output_dir, "demo_signals.jsonl")
    with open(signals_path, "w", encoding="utf-8") as f:
        for sig in signals:
            f.write(json.dumps(sig, ensure_ascii=False) + "\n")

    agg_output = os.path.join(output_dir, "aggregated.safetensors")
    agg_sha = fedavg_weighted(adapter_paths, signals, agg_output)

    # ── Step 6: Mock evaluation ──
    print("\n[6/7] 模拟评估 (关键词覆盖)...")
    holdout_path = PROJECT_ROOT / "data" / f"{domain}_holdout.jsonl"
    if not holdout_path.exists():
        holdout_path = Path(f"data/{domain}_holdout.jsonl")

    if holdout_path.exists():
        holdout_items = load_jsonl_tolerant(str(holdout_path))

        # Mock: keyword coverage (not real model eval)
        train_keywords = set()
        for item in all_items[:20]:
            output = item.get("output", "")
            for word in output.split()[:3]:
                if len(word) > 1:
                    train_keywords.add(word.lower())

        covered = 0
        for item in holdout_items:
            output = item.get("output", "")
            if any(kw in output.lower() for kw in train_keywords):
                covered += 1

        mock_acc = covered / len(holdout_items) if holdout_items else 0
        print(f"  Holdout: {len(holdout_items)} 条")
        print(f"  模拟准确率 (关键词覆盖): {mock_acc:.1%}")
        print(f"  注意: 真实评估需要 GPU + 模型加载")
        print(f"  真实评估: python scripts/eval_aggregated.py \\")
        print(f"    --adapter {agg_output} --domain {domain} --holdout {holdout_path}")
    else:
        print(f"  未找到 holdout 集: {holdout_path}")
        print("  跳过评估")

    # ── Step 7: Report ──
    print("\n" + "=" * 60)
    print("  Demo 报告")
    print("=" * 60)
    print(f"  领域: {domain}")
    print(f"  节点数: {num_nodes}")
    print(f"  训练步数: {steps}")
    print(f"  训练耗时: {train_time:.1f}s")
    print(f"  聚合 SHA: {agg_sha[:16]}...")
    print(f"  输出目录: {output_dir}")
    print()
    print("  各节点结果:")
    for nr in node_results:
        print(f"    Node {nr['node_id']}: loss={nr['final_loss']:.4f}, samples={nr['sample_count']}")
    print()
    print("  Pipeline 完成!")
    print(f"  聚合权重: {output_dir}/aggregated.safetensors")
    print(f"  下一步: firefly-node chat --adapter {output_dir}/aggregated.safetensors")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Firefly LM Demo Mode")
    parser.add_argument("--domain", default="law", help="领域 (law/medical/python/tax/education)")
    parser.add_argument("--nodes", type=int, default=4, help="模拟节点数 (默认 4)")
    parser.add_argument("--steps", type=int, default=5, help="每个节点训练步数 (默认 5)")
    parser.add_argument("--output", default="./demo_output", help="输出目录 (默认 ./demo_output)")
    args = parser.parse_args()

    success = run_demo(args.domain, args.nodes, args.steps, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
