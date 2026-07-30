# Day 3: FedAvg 聚合数学

> Day 3 课程 — 理解多个适配器怎么合并成一个更好的适配器

## 这节课你将学到什么

- FedAvg（联邦平均）的核心思想
- 等权平均 vs 软加权平均的区别
- 权重是怎么算出来的（公式拆解）
- 为什么"谁的贡献大，谁的权重就大"
- 亲手跑一次加权聚合

## 前置条件

- 已完成 Day 1（demo mode）和 Day 2（train-local）
- 有至少 2 个 `.safetensors` 适配器文件（demo 输出的或自己训练的）

## 1. 为什么需要聚合

Day 2 你训练了一个适配器。但联邦学习的核心是：**多个节点各训各的，最后合成一个**。

```
Node A (100 条法律数据) → adapterA
Node B (50 条法律数据)  → adapterB     ──→ 聚合 → 一个更好的 adapter
Node C (30 条法律数据)  → adapterC
```

为什么不直接用 Node A 的？因为 A 只有 100 条数据，可能没覆盖到 B 和 C 擅长的领域。聚合后的模型"见"过所有人的数据——但原始数据从未离开过各自的机器。

## 2. 等权平均（朴素 FedAvg）

最简单的做法：把三个适配器的权重张量直接取平均。

```
aggregated = (adapterA + adapterB + adapterC) / 3
```

**问题**：Node A 用 100 条数据认真训练了 30 步，Node C 只用 10 条数据随便训了 5 步。等权平均让 C 的"垃圾权重"和 A 的"好权重"一样重要，这不公平。

## 3. 软加权平均（火种用的方法）

火种的 `fedavg_weighted.py` 用三步算权重：

### 第 1 步：基础权重 = 样本数

```
Node A: 100 条 → weight_A = 100
Node B: 50 条  → weight_B = 50
Node C: 30 条  → weight_C = 30
```

样本越多，训练越充分，权重越大。

### 第 2 步：提升系数 = 效果奖励

```
improvement_factor = 1 + max(0, holdout_improvement)
```

`holdout_improvement` 是这个节点在 holdout 集上的准确率提升。如果节点 A 让 holdout 准确率提升了 5%（0.05），它的提升系数就是 1.05。如果没提升（0），系数就是 1.0。用 `max(0, ...)` 确保负数（越练越差）不会变成惩罚。

### 第 3 步：合成 + 归一化 + 裁剪

```
raw_weight = sample_count × improvement_factor
normalized = raw_weight / sum(all raw_weights)
clipped = clip(normalized, 0.05, 0.95)  → 重新归一化
```

裁剪到 `[0.05, 0.95]` 确保没有任何节点被完全忽略（最小 5%），也没有节点垄断聚合（最大 95%）。

### 完整例子

| 节点 | 样本数 | holdout 提升 | 原始权重 | 归一化后 |
|------|--------|-------------|---------|---------|
| A | 100 | 0.05 | 100 × 1.05 = 105 | 105/148 = 0.710 |
| B | 30 | 0.02 | 30 × 1.02 = 30.6 | 30.6/148 = 0.207 |
| C | 10 | 0.00 | 10 × 1.00 = 10 | 10/148 = 0.068 |

**总和** = 105 + 30.6 + 10 = 148

Node A 拿到 71% 的权重——它贡献最多，理应权重最大。但 B 和 C 也没被忽略（20.7% 和 6.8%），它们的数据虽然少，但可能覆盖了 A 没见过的知识点。

## 4. 实际张量层面发生了什么

LoRA 适配器是几个小矩阵（比如 `q_proj` 和 `v_proj` 的 `lora_A` 和 `lora_B`）。聚合就是把这些矩阵按权重加权求和：

```python
# 伪代码
for key in ["q_proj.lora_A", "q_proj.lora_B", "v_proj.lora_A", "v_proj.lora_B"]:
    stacked = torch.stack([adapterA[key], adapterB[key], adapterC[key]])  # 堆叠
    weight_tensor = torch.tensor([0.710, 0.207, 0.068])
    aggregated[key] = (stacked * weight_tensor).sum(dim=0)  # 加权求和
```

每个矩阵的每个元素都按同样的权重比例混合。

## 5. 动手跑一次

```bash
# 用 demo 输出的适配器
python scripts/fedavg_weighted.py \
  --adapters demo_output/node_0_adapter.safetensors,demo_output/node_1_adapter.safetensors,demo_output/node_2_adapter.safetensors \
  --signals demo_output/demo_signals.jsonl \
  --output demo_output/my_aggregated.safetensors
```

你会看到每个节点的权重计算过程和最终结果。

### 用信号文件控制权重

`demo_signals.jsonl` 长这样：

```json
{"task_id": "demo_law_node_0", "sample_count": 6, "holdout_improvement": 0.05, "final_loss": 0.45}
{"task_id": "demo_law_node_1", "sample_count": 6, "holdout_improvement": 0.03, "final_loss": 0.52}
```

试着改 `sample_count` 和 `holdout_improvement` 的值，看权重怎么变。

## 6. 等权 vs 软加权对比

| 维度 | 等权 FedAvg | 软加权 FedAvg |
|------|------------|--------------|
| 计算 | 简单平均 | 加权平均 |
| 公平性 | 贡献大的被稀释 | 贡献大的权重大 |
| 抗噪声 | 一个垃圾 adapter 拖累整体 | 垃圾 adapter 权重低 |
| 需要信号 | 不需要 | 需要 sample_count + holdout |
| 适用场景 | 节点均匀 | 节点异质（火种的场景） |

## 练习

| 练习 | 说明 |
|------|------|
| 改信号 | 修改 `demo_signals.jsonl` 里某个节点的 `sample_count`，看权重变化 |
| 消零实验 | 把某个节点的 `holdout_improvement` 设成负数，看它的权重会不会变小 |
| 等权对比 | 用等权平均 `--signals ""`（不传信号），对比聚合结果 |

## 常见问题

- **没有信号文件怎么办？** 不传 `--signals`，所有节点使用等权（1.0）。
- **权重全一样？** 检查信号文件路径是否正确，`task_id` 是否匹配。
- **裁剪为什么是 0.05？** 防止任何一个节点被完全忽略。可以用 `--min-weight` 调。

## 下一步

完成 Day 3 后，进入 Day 4：防作弊攻防——理解为什么联邦学习需要信任机制。
