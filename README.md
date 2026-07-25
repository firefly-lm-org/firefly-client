# Firefly LM Client (火种端)

> 分布式志愿算力节点的本地客户端。安装后贡献闲置 GPU 算力，参与社区 LoRA 持续微调。

🌐 官网：https://firefly-lm.com
💰 赞助：https://ifdian.net/a/firefly-lm

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/firefly-lm-org/firefly-client.git
cd firefly-client

# 2. 建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 装依赖
pip install -r requirements.txt

# 4. 跑通（无 GPU 用 mock）
firefly start --mock

# 5. 真实训练（需 ≥6GB VRAM）
export FIREFLY_MODEL_PATH="unsloth/Qwen3-1.5B-Instruct-4bit"
firefly start
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FIREFLY_MODEL_PATH` | `unsloth/Qwen3-1.5B-Instruct-4bit` | 底座模型路径或 HF repo |
| `FIREFLY_DATA_PATH` | （无） | 本地 JSONL/JSON 训练数据 |
| `FIREFLY_DATASET` | （无） | HuggingFace 数据集名 |
| `FIREFLY_LORA_RANK` | 32 | LoRA 秩 |
| `FIREFLY_LORA_ALPHA` | 64 | LoRA alpha |
| `FIREFLY_MAX_STEPS` | 100 | 训练步数 |
| `FIREFLY_BATCH_SIZE` | 2 | 每卡 batch |
| `FIREFLY_GRAD_ACCUM` | 4 | 梯度累积 |
| `FIREFLY_MAX_SEQ_LEN` | 2048 | 最大序列长度 |
| `FIREFLY_LR` | 2e-4 | 学习率 |

## 状态

v0.2 开发中 — 真实 QLoRA 训练骨架已就绪，等 GPU 机器验证。

## 许可证

Apache License 2.0
