# Day 4: 防作弊攻防

> Day 4 课程 — 理解为什么联邦学习需要信任机制，以及火种怎么检测作弊

## 这节课你将学到什么

- 联邦学习的信任假设是什么
- 三种常见的作弊方式
- cos 相似度检测原理（本地 L1）
- 服务器端轻量校验（L1 服务器侧）
- 双层防御架构：重检测在本地，轻校验在服务器

## 前置条件

- 已完成 Day 1-3
- 有 demo 输出的多个适配器文件

## 1. 信任假设

联邦学习建立在一个假设上：**每个节点诚实地训练了自己的数据**。

但如果没有防作弊，一个节点可以：

| 作弊方式 | 做法 | 后果 |
|---------|------|------|
| 偷懒复制 | 直接复制别人的适配器提交 | 白拿信誉分，没贡献新知识 |
| 随机噪声 | 提交随机生成的权重 | 稀释聚合效果 |
| 全零攻击 | 提交全零权重 | 把聚合结果拉向零，破坏模型 |

如果 4 个节点里 1 个作弊，聚合结果可能从 85% 准确率掉到 65%。

## 2. 双层防御架构

火种的防作弊分两层：

```
节点提交 adapter
      ↓
┌─────────────────────┐
│  本地 L1（重检测）    │  cos 相似度检测（cheat_detect.py）
│  在聚合前跑           │  能发现：复制/噪声/全零
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  服务器 L1（轻校验）   │  文件大小 + SHA256 + 频率
│  在 /tasks/complete 时 │  能发现：异常小文件/重复提交/高频提交
└─────────────────────┘
```

**为什么分两层？** 服务器是 Python 3.6.8，没有 torch/safetensors，做不了 cos 相似度计算。所以重检测放在本地（有 torch 的节点上跑），服务器只做轻量校验。

## 3. cos 相似度检测（本地 L1）

### 原理

两个向量越相似，cos 相似度越接近 1。越不同，越接近 0（甚至负数）。

```
adapter A 和 adapter B 的 cos 相似度 = 0.95  → 很可能 A 复制了 B
adapter A 和 adapter B 的 cos 相似度 = 0.02  → 正常，不同数据训出来的
```

### 火种怎么做

取第一个适配器作为基线，计算其他每个适配器与基线的平均 cos 相似度（跨所有张量 key）。

```bash
python scripts/cheat_detect.py \
  --adapters node_0_adapter.safetensors,node_1_adapter.safetensors,node_2_adapter.safetensors \
  --threshold 0.10
```

输出：

```
索引    相似度        状态
   0      1.0000      baseline
   1      0.0123      ok
   2     -0.0045      ok
   3      0.0089      ok

[OK] 全部 4 个 adapter 通过检测
```

如果某个适配器相似度 > 阈值（0.10），说明它和基线太像了，可能是复制的，会被标记为可疑，权重清零。

### 三种作弊的表现

| 作弊 | cos 相似度 | 为什么 |
|------|-----------|--------|
| 偷懒复制 | ≈ 1.0 | 完全一样的张量 |
| 随机噪声 | ≈ 0 | 和基线方向无关 |
| 全零攻击 | 0.0 | 零向量和任何向量正交 |

正常训练出来的适配器相似度通常在 -0.05 ~ 0.05 之间（不同数据、不同随机种子）。

## 4. 服务器端轻量校验（L1 服务器侧）

当节点调用 `/api/v1/tasks/complete` 提交结果时，服务器检查 4 项：

| 检查项 | 规则 | 处罚 |
|--------|------|------|
| final_loss 合理性 | 不能为负或零 | 信誉分 -50 |
| weight_path 文件大小 | < 1KB 可疑 | 信誉分 -50 |
| weight_hash 记录 | SHA256 前 16 位 | 记录 |
| 提交频率 | 5 分钟内 > 3 次 | 信誉分 -50 |

### 为什么这些检查有效

- **final_loss 为 0 或负**：真实训练的 loss 不可能为 0（除非模型完美过拟合），更不可能为负
- **文件 < 1KB**：真实 LoRA 适配器至少几十 KB，1KB 以下几乎肯定是空文件或垃圾
- **高频提交**：正常训练 30 步至少几分钟，5 分钟提交 3 次以上说明没在真正训练

### 信誉分机制

每个节点有信誉分，初始 100。触发检查项扣 50 分。信誉分低的节点：
- 优先级降低（分配任务时排在后面）
- 信誉分 < 30 可能被限制认领任务

## 5. 动手跑一次防作弊检测

```bash
# 用 demo 输出的适配器检测
python scripts/cheat_detect.py \
  --adapters demo_output/node_0_adapter.safetensors,demo_output/node_1_adapter.safetensors \
  --threshold 0.10 \
  --output demo_output/cheat_report.json
```

### 制造一个"作弊"场景

```bash
# 复制一个适配器（模拟偷懒）
cp demo_output/node_0_adapter.safetensors demo_output/fake_copy.safetensors

# 检测 — fake_copy 和 node_0 的相似度应该 ≈ 1.0
python scripts/cheat_detect.py \
  --adapters demo_output/node_0_adapter.safetensors,demo_output/fake_copy.safetensors \
  --threshold 0.10
```

你会看到 fake_copy 的相似度接近 1.0，被标记为可疑。

## 6. 完整使用链路

```
# 1. 聚合前先检测
python scripts/cheat_detect.py --adapters a.safetensors,b.safetensors --threshold 0.10

# 2. 通过检测后才聚合
python scripts/fedavg_weighted.py --adapters a.safetensors,b.safetensors --signals signals.jsonl --output agg.safetensors

# 3. 聚合后评估
python scripts/eval_aggregated.py --adapter agg.safetensors --domain law --holdout data/law_holdout.jsonl
```

防作弊在聚合**之前**跑——发现可疑适配器后，要么剔除，要么把它的权重清零，再进入聚合。

## 7. 未来：L2 检测

当前 L1 能抓"明显的"作弊。未来 L2 会加：

- **节点历史权重滑动均值偏离检测**：一个节点的适配器突然和之前的风格差很大
- **交叉验证**：同一任务分给 2 个节点，对比结果
- **零知识证明**：节点证明"我真的做了计算"，不泄露数据

这些需要更多节点和更多数据才能验证有效性，当前阶段不做。

## 练习

| 练习 | 说明 |
|------|------|
| 复制检测 | 复制一个适配器，看 cheat_detect 能不能抓到 |
| 阈值调参 | 把 `--threshold` 从 0.10 改到 0.01，看结果变化 |
| 读报告 | 看 `cheat_report.json` 里的结构化结果 |

## 常见问题

- **相似度为什么是负数？** 两个向量方向相反时 cos 为负。这不代表作弊，是正常现象。
- **阈值 0.10 怎么定的？** 经验值。低于 0.10 基本正常，高于 0.10 可能有问题。可以根据实际数据调。
- **服务器能做 cos 检测吗？** 当前不能（Python 3.6.8 无 torch），所以重检测在本地。

## 下一步

完成 Day 4 后，进入 Day 5：评估与优化——理解怎么用 holdout 集量化聚合效果。
