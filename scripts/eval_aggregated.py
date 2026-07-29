# -*- coding: utf-8 -*-
"""
聚合权重评估脚本
对每个领域跑固定 holdout 集，输出准确率并保存到 benchmarks/agg_score.json

用法：
  python scripts/eval_aggregated.py \
    --adapter outputs/aggregated/adapter_model.safetensors \
    --domain law \
    --holdout data/law_holdout.jsonl \
    --output benchmarks/agg_score.json

评估规则：
  - 加载 base model + adapter
  - 对 holdout 集每道题做 generate
  - 预测输出包含参考答案关键词即为正确
  - 结果追加保存到 benchmarks/agg_score.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except Exception as e:
    print(f"[eval_aggregated] 缺少依赖: {e}")
    print("请先安装: pip install torch transformers peft")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="聚合权重评估")
    parser.add_argument("--adapter", type=str, required=True,
                        help="聚合权重路径")
    parser.add_argument("--domain", type=str, required=True,
                        help="领域名称（law/medical/python/tax/education）")
    parser.add_argument("--holdout", type=str, required=True,
                        help="holdout 集路径（JSONL）")
    parser.add_argument("--output", type=str, default="benchmarks/agg_score.json",
                        help="评估结果输出路径")
    parser.add_argument("--base-model", type=str,
                        default="unsloth/Qwen3-1.5B-Instruct-4bit",
                        help="基础模型名称")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="最大生成 token 数")
    return parser.parse_args()


def load_holdout(path):
    """加载 holdout 集"""
    questions = []
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                questions.append(item.get("instruction", ""))
                answers.append(item.get("output", ""))
    return questions, answers


def evaluate(adapter_path, base_model_name, questions, answers, max_new_tokens):
    """加载 adapter 并评估"""
    print(f"加载基础模型: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"加载 adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    correct = 0
    total = len(questions)
    results = []

    for i, (q, a) in enumerate(zip(questions, answers)):
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        pred = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # 简单匹配：预测包含参考答案的关键词即为正确
        keywords = a.split()[:5]  # 取前 5 个词作为关键词
        is_correct = any(kw.lower() in pred.lower() for kw in keywords)

        if is_correct:
            correct += 1

        results.append({
            "question": q,
            "expected": a[:100],
            "predicted": pred[:100],
            "correct": is_correct,
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{total}] 当前准确率: {correct/(i+1):.2%}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, results


def save_results(domain, adapter_path, accuracy, results, output_path):
    """保存评估结果"""
    # 计算 adapter SHA256
    sha = hashlib.sha256()
    with open(adapter_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    score_entry = {
        "domain": domain,
        "adapter": adapter_path,
        "adapter_sha256": sha.hexdigest(),
        "accuracy": accuracy,
        "num_questions": len(results),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # 读取已有结果
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
    else:
        scores = []

    # 追加新结果
    scores.append(score_entry)

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存: {output_path}")
    print(f"领域: {domain}")
    print(f"准确率: {accuracy:.2%} ({score_entry['num_questions']} 题)")
    print(f"Adapter SHA256: {sha.hexdigest()[:16]}...")


def main():
    args = parse_args()

    if not os.path.exists(args.adapter):
        print(f"错误: adapter 不存在: {args.adapter}")
        sys.exit(1)

    if not os.path.exists(args.holdout):
        print(f"错误: holdout 集不存在: {args.holdout}")
        sys.exit(1)

    print(f"评估聚合权重: {args.adapter}")
    print(f"领域: {args.domain}")
    print(f"Holdout 集: {args.holdout}")

    questions, answers = load_holdout(args.holdout)
    print(f"加载 holdout: {len(questions)} 题")

    accuracy, results = evaluate(
        args.adapter, args.base_model,
        questions, answers, args.max_new_tokens,
    )

    save_results(args.domain, args.adapter, accuracy, results, args.output)


if __name__ == "__main__":
    main()
