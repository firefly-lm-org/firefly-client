# Day 2: 训练你的第一个适配器

## 目标

用自己的数据训练一个 LoRA 适配器，理解微调的基本原理。

## 前置条件

- 已完成 Day 1（demo mode 跑通）
- 有 GPU（RTX 3060 或更好）或愿意用 CPU 慢速训练
- `pip install -e .` + `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## 数据准备

Firefly 使用 JSONL 格式，每行一个 `{"instruction": "...", "output": "..."}` 对。

```json
{"instruction": "什么是合同法的诚实信用原则？", "output": "诚实信用原则要求当事人在民事活动中..."}
{"instruction": "违约金的计算标准是什么？", "output": "违约金的计算以实际损失为基础..."}
```

### 用自带数据

```bash
# 法律领域已有 72 条
firefly-node train-local --dataset data/law_qa.jsonl --domain law --output my_adapter.safetensors --steps 30
```

### 用自己的数据

创建 `my_data.jsonl`，至少 20 条：

```bash
firefly-node train-local --dataset my_data.jsonl --domain law --output my_adapter.safetensors --steps 30
```

## 训练过程

训练时你会看到 loss 逐步下降：

```
Step 1/30: loss=2.4567
Step 5/30: loss=1.8234
Step 10/30: loss=1.4567
Step 15/30: loss=1.1234
Step 20/30: loss=0.9876
Step 30/30: loss=0.8234
训练完成! final_loss=0.8234
适配器已保存: my_adapter.safetensors
```

**关注点**：loss 从 ~2.5 降到 ~0.8 说明模型在学习。如果 loss 不降反升，可能学习率太高。

## 验证效果

```bash
# 用训练好的适配器聊天
firefly-node chat --adapter my_adapter.safetensors

# 输入一条法律问题，看模型回答
> 合同违约后如何计算赔偿？
```

## 练习

| 练习 | 说明 |
|------|------|
| 步数对比 | 分别跑 `--steps 10` 和 `--steps 30`，比较 chat 效果 |
| 域切换 | 用 `data/education_qa.jsonl` 训练教育适配器 |
| 自己的数据 | 写 20 条你专业领域的数据，训练专属适配器 |

## 常见问题

- **训练很慢？** CPU 模式下 30 步约 2 小时，RTX 3060 约 30 分钟。demo mode 不需要 GPU。
- **OOM 显存不足？** 减小 `--steps` 或用更小的 base model。
- **loss 不降？** 检查数据格式，确保 instruction 和 output 都不为空。

## 下一步

完成 Day 2 后，进入 Day 3：理解 FedAvg 聚合数学（`docs/demo_guide.md` 第 5 步详解）。
