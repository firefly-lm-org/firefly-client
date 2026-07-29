# Demo Mode: 30 分钟跑通联邦 LLM 训练管线

> Day 1 课程 — 面向刚 clone 仓库的学生

## 这节课你将学到什么

- 联邦学习（Federated Learning）的基本概念
- 如何用一条命令在单机上模拟 4 节点联邦训练
- FedAvg 加权聚合的原理
- 防作弊检测为什么重要
- 如何评估聚合效果

## 1. 什么是联邦学习

传统机器学习：把所有数据集中到一个地方训练。

联邦学习：每个节点在自己的数据上训练，只交换模型权重（LoRA adapter），不交换原始数据。

```
传统:  [数据A] + [数据B] + [数据C] → 集中训练 → 模型
联邦:  [数据A] → 训练 → adapterA ─┐
       [数据B] → 训练 → adapterB ─┼→ 聚合 → 模型
       [数据C] → 训练 → adapterC ─┘
```

**为什么需要联邦学习？**

- 数据隐私：医院不能共享病人数据，律所不能共享案件文档
- 数据主权：每个机构保留自己的数据，只分享"学到的知识"（权重）
- 分布式算力：利用多台机器的 GPU，不需要集中到一处

## 2. 运行 Demo

### 前置条件

```bash
git clone --depth 1 https://github.com/firefly-lm-org/firefly-client.git
cd firefly-client

pip install -e .
pip install torch safetensors
```

> 不需要 GPU。Demo 使用 mock 训练（随机 LoRA 张量），在 CPU 上 30 秒内跑完。

### 一条命令

```bash
python scripts/demo_mode.py
```

或通过 CLI：

```bash
firefly-node demo
```

### 自定义参数

```bash
python scripts/demo_mode.py --domain law --nodes 4 --steps 5 --output ./demo_output
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | law | 领域 (law/medical/python/tax/education) |
| `--nodes` | 4 | 模拟节点数 |
| `--steps` | 5 | 每个节点的训练步数 |
| `--output` | ./demo_output | 输出目录 |

## 3. Demo 做了什么（7 步管线）

### Step 1: 数据切分

把 `data/law_qa.jsonl` 按节点数平分。每个节点拿到不同的子集。

```
23 条法律 QA → 4 个节点
  Node 0: 6 条 → node_0_data.jsonl
  Node 1: 6 条 → node_1_data.jsonl
  Node 2: 6 条 → node_2_data.jsonl
  Node 3: 5 条 → node_3_data.jsonl
```

**关键点**：数据不交换。每个节点只看自己的数据。

### Step 2: 模拟训练

每个节点用不同的随机种子生成 mock LoRA adapter（.safetensors 文件）。

- 4 个张量：q_proj 和 v_proj 的 lora_A / lora_B 矩阵
- shape: (8, 256) 和 (256, 8) — 比真实 LoRA 小，但结构一致
- 每个节点的张量值不同（不同随机种子）
- 模拟 loss 递减

> **为什么是 mock？** 真实 QLoRA 训练需要 GPU + unsloth + 30 分钟。Demo 用随机张量替代，让你先看懂流程。真实训练见 `firefly-node train-local`。

### Step 3: 模拟联邦流程

模拟每个节点执行 `claim → train → complete`：

```
Node 0: claim → train → complete OK
Node 1: claim → train → complete OK
Node 2: claim → train → complete OK
Node 3: claim → train → complete OK
```

**claim**：节点向调度中心申请任务
**train**：节点在本地数据上训练
**complete**：节点提交训练结果（loss、adapter 路径、SHA256）

### Step 4: 防作弊检测

聚合前，检测各 adapter 与基线（第一个 adapter）的 cos 相似度。

```
索引    相似度        状态
   0      1.0000      baseline
   1      0.0123      ok
   2     -0.0045      ok
   3      0.0089      ok
```

**为什么要检测？**

联邦学习的信任假设：每个节点诚实地训练了自己的数据。但如果一个节点：
- 直接复制别人的 adapter（偷懒）
- 提交随机噪声（恶意）
- 提交全零权重（破坏聚合）

cos 相似度能发现这些异常。低于阈值的 adapter 权重清零，不参与聚合。

### Step 5: FedAvg 加权聚合

把多个 adapter 按权重合并成一个。

**权重计算**：
```
基础权重 = sample_count（样本越多，权重越大）
提升系数 = 1 + max(0, holdout_improvement)（效果越好，权重越大）
最终权重 = 基础权重 × 提升系数
归一化后裁剪到 [0.05, 0.95]
```

**为什么不是等权平均？**

- 节点 A 用 100 条数据训练，节点 B 用 10 条 → A 的权重应该更大
- 节点 A 的 holdout 准确率提升 5%，节点 B 提升 0% → A 的权重应该更大
- 这叫**软加权 FedAvg**，比等权平均更公平

### Step 6: 模拟评估

对 holdout 集做关键词覆盖检测（mock）。

> 真实评估需要加载 base model + adapter 做推理。Demo 跳过这一步，但告诉你怎么跑：
> ```bash
> python scripts/eval_aggregated.py \
>   --adapter demo_output/aggregated.safetensors \
>   --domain law \
>   --holdout data/law_holdout.jsonl
> ```

### Step 7: 报告

输出每个节点的 loss、样本数、聚合 SHA256。

## 4. 输出文件

```
demo_output/
  node_0_data.jsonl          # Node 0 的训练数据子集
  node_1_data.jsonl          # Node 1 的训练数据子集
  ...
  node_0_adapter.safetensors # Node 0 的 mock LoRA adapter
  node_1_adapter.safetensors # Node 1 的 mock LoRA adapter
  ...
  demo_signals.jsonl          # 训练信号（loss/sample_count/holdout_improvement）
  aggregated.safetensors      # FedAvg 聚合后的 adapter
```

## 5. 下一步

| 想做什么 | 命令 |
|---------|------|
| 真实训练（需 GPU） | `firefly-node train-local --dataset data/law_qa.jsonl --domain law --steps 30` |
| 推理验证 | `firefly-node chat --adapter demo_output/aggregated.safetensors` |
| 连接调度中心 | `firefly-node fed status` |
| 认领真实任务 | `firefly-node fed claim --domain law` |
| 手动聚合 | `python scripts/fedavg_weighted.py --adapters a.safetensors,b.safetensors --signals signals.jsonl --output agg.safetensors` |
| 真实评估 | `python scripts/eval_aggregated.py --adapter agg.safetensors --domain law --holdout data/law_holdout.jsonl` |
| 防作弊检测 | `python scripts/cheat_detect.py --adapters a.safetensors,b.safetensors --threshold 0.10` |

## 6. 概念速查

| 术语 | 解释 |
|------|------|
| LoRA adapter | 小型权重补丁，加在 base model 上改变其行为 |
| FedAvg | 把多个 adapter 按权重平均，合成一个 |
| Holdout | 不参与训练的测试题，用来验证聚合效果 |
| 信号 (signal) | 节点提交的训练元数据（loss、样本数、holdout 改善） |
| 防作弊 L1 | cos 相似度检测，发现可疑 adapter |
| SHA256 | adapter 文件的指纹，用于追溯和防篡改 |
