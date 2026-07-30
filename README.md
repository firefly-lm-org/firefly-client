# Firefly LM Client (火种端)

> 300 块月费维持的调度中心，协调全球志愿算力做持续垂直微调。代码免费，服务收费；本机训练，权重归你。

🌐 官网：http://106.14.220.169/
💰 赞助：https://afdian.com/a/firefly-lm
📦 GitHub：https://github.com/firefly-lm-org/firefly-client

## 快速开始

```bash
# 1. 克隆
git clone --depth 1 https://github.com/firefly-lm-org/firefly-client.git
cd firefly-client

# 2. 安装 CLI（不含 torch）
pip install -e .

# 3. 安装 GPU 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets sentencepiece unsloth \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 设置调度中心地址
# Linux/Mac:  echo '{"server_url":"http://106.14.220.169:8000"}' > ~/.firefly/config.json
# Windows:    创建 %USERPROFILE%\.firefly\config.json，内容: {"server_url":"http://106.14.220.169:8000"}

# 5. 本地训练（约 5 分钟）
curl -LO https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/law_qa.jsonl
firefly-node train-local --dataset law_qa.jsonl --domain law \
    --output my_adapter.safetensors --steps 30

# 6. 推理验证
firefly-node chat --adapter my_adapter.safetensors \
    --prompt "劳动合同到期不续签有赔偿吗？"
```

## 命令一览

| 命令 | 说明 |
|------|------|
| `firefly-node doctor` | 环境检查（Python/torch/配置/服务器/数据，5秒验证） |
| `firefly-node demo` | 单机模拟 4 节点联邦训练（CPU，30秒跑完） |
| `firefly-node train-local` | 本地 QLoRA 训练（数据不出本机） |
| `firefly-node chat` | 用 LoRA 适配器推理 |
| `firefly-node verify` | 验证适配器来源（SHA256 manifest 比对） |
| `firefly-node fed status` | 查看调度中心状态 |
| `firefly-node fed claim` | 认领联邦训练任务 |
| `firefly-node fed train` | 执行联邦训练 |
| `firefly-node fed complete` | 回传训练结果（脱敏信号） |
| `firefly-node fed download` | 下载聚合权重 |
| `firefly-node register` | 注册用户 |
| `firefly-node login` | 登录 |
| `firefly-node start` | 开始贡献算力（后台） |
| `firefly-node status` | 查看节点状态 |

## 5 天课程

| 天 | 主题 | 文档 |
|----|------|------|
| 1 | 联邦学习概念 + demo mode | [docs/demo_guide.md](docs/demo_guide.md) |
| 2 | 训练你的第一个适配器 | [docs/day2_train_local.md](docs/day2_train_local.md) |
| 3 | FedAvg 聚合数学 | [docs/day3_fedavg_math.md](docs/day3_fedavg_math.md) |
| 4 | 防作弊攻防 | [docs/day4_anticheat.md](docs/day4_anticheat.md) |
| 5 | 评估与优化 | [docs/day5_evaluation.md](docs/day5_evaluation.md) |

## 常用脚本

| 脚本 | 说明 |
|------|------|
| `scripts/demo_mode.py` | 单机多节点联邦训练模拟 |
| `scripts/fedavg_weighted.py` | 软加权 FedAvg 聚合 |
| `scripts/eval_aggregated.py` | 聚合后 holdout 评估 |
| `scripts/cheat_detect.py` | 本地 cos 相似度防作弊检测 |

## 文档

| 文档 | 说明 |
|------|------|
| [FAQ](docs/faq.md) | 常见问题 |
| [学生创业指南](docs/student_entrepreneur_guide.md) | 用火种做你的第一个 AI 产品 |
| [学生创业合作政策](docs/student_partner_policy.md) | 合作模式与红线 |
| [外部测试邀请](docs/external_test_invitation.md) | 测试者指南 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FIREFLY_MODEL_PATH` | `unsloth/Qwen3-1.5B-Instruct-4bit` | 底座模型 |
| `FIREFLY_MAX_STEPS` | 60 | 训练步数 |
| `FIREFLY_LORA_RANK` | 8 | LoRA 秩 |
| `FIREFLY_LORA_ALPHA` | 16 | LoRA alpha |
| `FIREFLY_BATCH_SIZE` | 1 | 每卡 batch |
| `FIREFLY_GRAD_ACCUM` | 4 | 梯度累积 |
| `FIREFLY_MAX_SEQ_LENGTH` | 512 | 最大序列长度 |
| `FIREFLY_LR` | 2e-4 | 学习率 |

## 状态

v0.6 已发布 — 本地训练 + 联邦命令组 + 调度中心 19 端点在线。

## 许可证

MIT License
