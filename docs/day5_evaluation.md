# Day 5: 评估与优化

> Day 5 课程 — 理解怎么量化聚合效果，以及怎么让模型越来越好

## 这节课你将学到什么

- 什么是 holdout 集以及为什么需要它
- 怎么跑评估脚本看准确率
- `agg_score.json` 是什么，怎么用它跟踪进步
- 聚合 vs 单节点的效果对比
- 5 个优化方向

## 前置条件

- 已完成 Day 1-4
- 有一个聚合后的适配器（demo 输出的或自己聚合的）
- 有 holdout 集（`data/law_holdout.jsonl`）

## 1. 什么是 holdout 集

训练时用的数据叫训练集。但我们不能用训练集来评估效果——模型可能只是"背"下了答案。

**holdout 集** = 模型从没见过的题。只有 holdout 准确率高，才说明模型真的"学"到了知识，而不是死记硬背。

```
训练集: 23 条法律 QA → 训练 → adapter
holdout 集: 5 条法律 QA → 评估 → "答对了 4 条，准确率 80%"
```

火种的 holdout 集在 `data/law_holdout.jsonl`，5 道题，用固定 seed=42 生成，确保可复现。

## 2. 跑评估

```bash
python scripts/eval_aggregated.py \
  --adapter demo_output/aggregated.safetensors \
  --domain law \
  --holdout data/law_holdout.jsonl \
  --output benchmarks/agg_score.json
```

### 评估过程

1. 加载基础模型（Qwen3-1.5B）+ 聚合适配器
2. 对 holdout 集每道题做推理（generate）
3. 检查预测输出是否包含参考答案的关键词
4. 算准确率 = 答对数 / 总题数
5. 结果追加保存到 `benchmarks/agg_score.json`

### 输出示例

```
加载基础模型: unsloth/Qwen3-1.5B-Instruct-4bit
加载 adapter: demo_output/aggregated.safetensors
  [1/5] 当前准确率: 20.00%
  [5/5] 当前准确率: 60.00%

评估结果已保存: benchmarks/agg_score.json
领域: law
准确率: 60.00% (5 题)
Adapter SHA256: a1b2c3d4e5f67890...
```

> **注意**：真实评估需要 GPU + transformers + peft。demo mode 用的是 mock 张量，评估结果不代表真实效果。Day 5 的核心是理解评估流程和怎么看结果。

## 3. agg_score.json 怎么用

每次评估的结果**追加**到同一个文件。跑 3 次后，文件长这样：

```json
[
  {
    "domain": "law",
    "adapter": "outputs/r1/aggregated.safetensors",
    "adapter_sha256": "a1b2c3...",
    "accuracy": 0.40,
    "num_questions": 5,
    "timestamp": "2026-07-29T10:00:00"
  },
  {
    "domain": "law",
    "adapter": "outputs/r2/aggregated.safetensors",
    "adapter_sha256": "b2c3d4...",
    "accuracy": 0.60,
    "num_questions": 5,
    "timestamp": "2026-07-29T12:00:00"
  },
  {
    "domain": "law",
    "adapter": "outputs/r3/aggregated.safetensors",
    "adapter_sha256": "c3d4e5...",
    "accuracy": 0.80,
    "num_questions": 5,
    "timestamp": "2026-07-29T14:00:00"
  }
]
```

### 看趋势

```bash
python -c "
import json
scores = json.load(open('benchmarks/agg_score.json'))
for s in scores:
    print(f'{s[\"timestamp\"][:10]}  {s[\"accuracy\"]:.0%}  {s[\"adapter_sha256\"][:8]}')
"
```

输出：
```
2026-07-29  40%  a1b2c3d4
2026-07-29  60%  b2c3d4e5
2026-07-29  80%  c3d4e5f6
```

准确率从 40% → 60% → 80%，说明每一轮聚合都在进步。

## 4. 聚合 vs 单节点效果对比

这是评估的核心问题：**聚合比单节点训练好吗？**

```bash
# 评估单节点
python scripts/eval_aggregated.py \
  --adapter demo_output/node_0_adapter.safetensors \
  --domain law --holdout data/law_holdout.jsonl \
  --output benchmarks/single_score.json

# 评估聚合
python scripts/eval_aggregated.py \
  --adapter demo_output/aggregated.safetensors \
  --domain law --holdout data/law_holdout.jsonl \
  --output benchmarks/agg_score.json

# 对比
python -c "
import json
single = json.load(open('benchmarks/single_score.json'))[-1]
agg = json.load(open('benchmarks/agg_score.json'))[-1]
print(f'单节点: {single[\"accuracy\"]:.0%}')
print(f'聚合后: {agg[\"accuracy\"]:.0%}')
print(f'提升: {(agg[\"accuracy\"] - single[\"accuracy\"]):+.0%}')
"
```

如果聚合准确率 > 单节点平均，说明联邦有效。

## 5. 五个优化方向

| # | 方向 | 怎么做 | 预期效果 |
|---|------|--------|---------|
| 1 | 增加数据 | 每个节点从 20 条加到 50 条 | 准确率 +5~10% |
| 2 | 增加步数 | `--steps 30` → `--steps 60` | 准确率 +5~15% |
| 3 | 增加节点 | 4 节点 → 8 节点 | 数据多样性提升 |
| 4 | 调学习率 | `--lr 2e-4` → `--lr 1e-4`（更稳） | loss 更平滑 |
| 5 | 换域 | law → education_math | 不同域效果不同 |

### 优化循环

```
改参数 → 训练 → 聚合 → 评估 → 看 agg_score.json 趋势
  ↑                                              |
  └────────────── 没变好就回退 ←─────────────────┘
```

每次只改一个变量，跑评估，看趋势。`agg_score.json` 是你的实验日志。

## 6. holdout 集怎么设计

好的 holdout 集应该：

| 原则 | 说明 |
|------|------|
| 不和训练集重叠 | holdout 的题不能出现在训练数据里 |
| 覆盖核心知识点 | 5 题覆盖 5 个不同法律概念 |
| 有明确答案 | 答案是客观的，不是主观评价 |
| 数量适中 | 太少不统计显著，太多评估慢。5-20 题合适 |

火种的 `data/law_holdout.jsonl` 5 道题覆盖：劳动纠纷、合同违约、侵权责任、知识产权、婚姻财产。

## 7. 验证适配器来源

评估前可以用 verify 确认适配器没被篡改：

```bash
firefly-node verify --adapter demo_output/aggregated.safetensors
```

如果有 manifest 文件，会比对 SHA256。确认通过后再跑评估，确保你评估的是"真正的"聚合结果。

## 练习

| 练习 | 说明 |
|------|------|
| 单 vs 聚 | 评估单节点和聚合，对比准确率 |
| 步数实验 | steps=10 vs steps=30，看 agg_score 趋势 |
| 写 holdout | 给教育领域写 5 题 holdout，评估教育适配器 |
| 看 SHA | 每次 SHA256 不同，确认每次聚合都是独立的 |

## 常见问题

- **准确率 0%？** demo mode 用的是 mock 张量，不是真实训练。用 `train-local` 训练后评估才有意义。
- **评估很慢？** 加载模型 + 逐题推理，5 题约 2-5 分钟（GPU）。CPU 会更慢。
- **关键词匹配太简单？** 当前用"预测包含参考答案前 5 个词"判断。未来可以换成 LLM 评分或语义匹配。

## 5 天课程回顾

| 天 | 主题 | 文档 |
|----|------|------|
| 1 | 联邦学习概念 + demo | `docs/demo_guide.md` |
| 2 | 训练第一个适配器 | `docs/day2_train_local.md` |
| 3 | FedAvg 聚合数学 | `docs/day3_fedavg_math.md` |
| 4 | 防作弊攻防 | `docs/day4_anticheat.md` |
| 5 | 评估与优化 | `docs/day5_evaluation.md` |

完成 5 天后，你应该能：独立训练适配器 → 理解聚合原理 → 检测作弊 → 评估效果 → 持续优化。

## 下一步

- [学生创业指南](student_entrepreneur_guide.md) — 把你的适配器变成产品
- [学生创业合作政策](student_partner_policy.md) — 了解合作模式
- [FAQ](faq.md) — 常见问题
- 加入社区：GitHub Issues 提问
