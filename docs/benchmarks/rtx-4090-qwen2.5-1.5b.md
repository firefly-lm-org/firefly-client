# RTX 4090 QLoRA Benchmark

**Date**: 2026-07-26
**Hardware**: NVIDIA RTX 4090 24GB (AutoDL)
**Model**: Qwen2.5-1.5B-Instruct
**LoRA Config**: r=8, alpha=16, targets=[q_proj, v_proj]
**Batch**: 1, GradAccum: 4, LR: 2e-4, max_steps: 60
**Dataset**: 29 samples (firefly alpaca_demo.jsonl)
**Training time**: 51.4s
**Final train loss**: 0.8754
**Trainable params**: 1,089,536 / 1,544,803,840 (0.0705%)
**GPU memory**: 1.07 GB peak
**Key fix**: accelerate big_modeling.py dispatch_model patch (bitsandbytes 0.50.0 + transformers 4.44.0)
