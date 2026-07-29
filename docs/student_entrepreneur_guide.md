# 学生创业指南：用火种做你的第一个 AI 产品

> 你学完了联邦学习，下一步是什么？把它变成产品。

## 你能做什么

### 路径 1：垂直领域辅导 LoRA 商店

训练一个垂直领域 LoRA 适配器，挂在网上卖。

| 适配器类型 | 目标用户 | 建议定价 |
|-----------|---------|---------|
| 高考数学辅导 | 高中生/家长 | ¥9.9 - ¥29.9 |
| 雅思写作批改 | 留学生 | ¥19.9 - ¥49.9 |
| 法律咨询助手 | 法律从业者 | ¥29.9 - ¥49.9 |
| Python 代码辅导 | 编程初学者 | ¥9.9 - ¥19.9 |

买家只需要 `transformers + peft` 就能用你的适配器，不需要装火种。

**真实类比**：Civitai 上卖 Stable Diffusion LoRA 的人月入几百到几千美元。LLM LoRA 市场还没起来，你是第一批。

### 路径 2：机构私有联邦部署

帮培训机构/学校搭火种联邦系统：

- 机构内 5 台电脑各自训自己的题库
- 火种聚合出统一的"机构辅导模型"
- 数据不出机构，合规无忧

| 服务 | 定价 |
|------|------|
| 部署 + 培训 | ¥2,000 - ¥5,000/次 |
| 月度维护 | ¥500 - ¥1,000/月 |
| 定制适配器训练 | ¥500 - ¥2,000/个 |

**真实需求**：新东方/好未来的分校之间有竞争关系，不愿意共享数据，但愿意共享模型能力。火种正好解决这个矛盾。

### 路径 3：学术论文 + 毕业设计

- 题目示例：《基于联邦学习的垂直领域 LoRA 聚合方法研究》
- 用火种跑实验：对比等权 FedAvg vs 软加权 vs 防作弊
- 数据来自教育/法律/医疗公开数据集
- 可发普刊或会议论文

**优势**：火种是现成可用的实验平台，不需要从零搭联邦学习环境。

## 你需要什么

| 条件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | RTX 3060 6GB | RTX 4090 24GB |
| 无 GPU？ | 租 AutoDL ¥1.5/h | — |
| 数据 | 20 条 JSONL | 200+ 条 |
| 火种 | 已安装 | `pip install -e .` |

## 操作步骤

### 1. 选择领域

```bash
# 查看已有数据格式
cat data/education_qa.jsonl | head -3
```

支持领域：`education` / `law` / `medical` / `python` / `tax`

### 2. 准备数据

创建你的数据集（JSONL 格式，每行一条）：

```json
{"instruction": "解释什么是勾股定理", "output": "勾股定理指出..."}
{"instruction": "计算 3x + 5 = 20 中 x 的值", "output": "x = 5..."}
```

最少 20 条，越多越好。数据质量决定适配器质量。

### 3. 训练适配器

```bash
firefly-node train-local \
  --dataset my_data.jsonl \
  --domain education \
  --output my_product.safetensors \
  --steps 60
```

60 步约 30 分钟（RTX 3060）。

### 4. 验证效果

```bash
# 交互式测试
firefly-node chat --adapter my_product.safetensors

# 验证来源（如果经过火种聚合）
firefly-node verify --adapter my_product.safetensors
```

### 5. 发布

把 `my_product.safetensors` 挂在平台上：

| 平台 | 优势 | 抽成 |
|------|------|------|
| 爱发电 | 国内创作者友好 | ~5% |
| 淘宝 | 流量大 | ~3% |
| 知识星球 | 社区属性强 | 平台费 |
| GitHub Releases | 免费，技术圈 | 0% |

买家拿到 `.safetensors` 文件后：
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-1.5B-Instruct-4bit")
model = PeftModel.from_pretrained(model, "my_product.safetensors")
```

### 6. 持续迭代

- 收集用户反馈，补充数据
- 增加训练步数提升效果
- 发布 v2、v3 版本
- 多个适配器打包卖（¥49.9 - ¥99.9）

## 定价策略

| 策略 | 适用场景 | 价格区间 |
|------|---------|---------|
| 免费引流版 | 5-10 步训练，效果一般 | ¥0 |
| 基础版 | 30-60 步训练 | ¥9.9 - ¥19.9 |
| 专业版 | 100+ 步，多轮优化 | ¥29.9 - ¥49.9 |
| 捆绑包 | 多领域打包 | ¥49.9 - ¥99.9 |
| 机构定制 | 联邦部署 | ¥2,000 - ¥5,000/次 |

## 注意事项

1. **数据合规**：不要在数据中包含个人信息（姓名、电话、身份证号等）
2. **适配器是标准格式**：买家不需要火种也能用，只要有 `transformers + peft`
3. **效果可验证**：用 `firefly-node verify` 证明适配器来源可信
4. **口碑 = 生命线**：效果越好，评价越高，定价可以越高
5. **开源 vs 闭源**：你可以开源适配器赚声誉，也可以闭源卖钱，火种不强制

## 从哪里开始

```bash
# 1. 跑 demo 看看效果
firefly-node demo

# 2. 看 Day 1 和 Day 2 文档
# docs/demo_guide.md
# docs/day2_train_local.md

# 3. 准备你的数据
# 20 条以上 JSONL

# 4. 开始训练
firefly-node train-local --dataset my_data.jsonl --domain education --output my_product.safetensors --steps 60

# 5. 测试效果
firefly-node chat --adapter my_product.safetensors

# 6. 发布
# 挂到爱发电/淘宝/知识星球
```

## 相关文档

- [学生创业合作政策](student_partner_policy.md) — 合作模式、角色定位、红线
- [Demo Guide](demo_guide.md) — Day 1 课程
- [Day 2](day2_train_local.md) — 训练你的第一个适配器
- [FAQ](faq.md) — 常见问题

## 一句话

火种给你的不是"一个工具"，是一条从学习到变现的路径：学联邦学习 → 训练垂直 LoRA → 挂网卖适配器 → 帮机构部署联邦。35MB 的适配器文件就是你的产品。
