# RTX 4090 QLoRA Benchmark

**Date**: 2026-07-26
**Hardware**: NVIDIA RTX 4090 24GB (AutoDL)
**Model**: Qwen2.5-1.5B-Instruct
**Quantization**: 4-bit NF4 (bitsandbytes)
**LoRA Config**: r=8, alpha=16, targets=[q_proj, v_proj]
**Batch**: 1, GradAccum: 4, LR: 2e-4, max_steps: 60
**Dataset**: 29 samples (firefly alpaca_demo.jsonl)
**Training time**: 51.4s
**Final train loss**: 0.8754
**Trainable params**: 1,089,536 / 1,544,803,840 (0.0705%)
**GPU memory peak**: 1.07 GB
**Status**: PASS

## Key Fixes Required

- **accelerate patch**: `bitsandbytes 0.50.0` + `transformers 4.44.0` + `accelerate 1.14.0` had a known incompatibility where `dispatch_model` calls `.to()` on a 4-bit quantized model, causing `ValueError: .to is not supported for 4-bit bitsandbytes models`. Fixed by patching `accelerate/big_modeling.py` to skip `.to()` for quantized models.
- **SFTTrainer packing**: Added `formatting_func` + removed explicit `DataCollatorForLanguageModeling` to avoid `ValueError: too many dimensions 'str'`.

## Training Log (step losses)

| Step | Loss |
|------|------|
| 10   | 1.1723 |
| 20   | 1.0029 |
| 30   | 0.8996 |
| 40   | 0.7846 |
| 50   | 0.7134 |
| 60   | 0.6799 |
