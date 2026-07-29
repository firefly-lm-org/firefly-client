# Firefly Node · 萤火虫火种客户端

> 基于开源底座的分布式联邦微调客户端 · 数据不出本机，只聚 LoRA 权重

## 功能

- 🔐 **安全登录** — JWT 认证，token 本地加密存储
- 🔧 **硬件自动检测** — CPU / 内存 / GPU / 显存自动识别
- ⚡ **智能断点** — 30 秒间隔，离线自动恢复
- 📥 **任务自动下载** — 从 MinIO 高速下载训练数据
- 🏋️ **训练模拟执行** — 进度可视化，资源占用可控
- 📤 **结果自动上传** — 完成后自动提交到调度中心
- 📊 **状态查询** — 实时查看节点状态、信誉分、任务进度

## 快速开始

### 安装

```bash
git clone git@github.com:firefly-lm-org/firefly-client.git
cd firefly-client
pip install -r requirements.txt
```

### 联邦训练（连调度中心）

```bash
firefly-node fed start \
  --domain law \
  --dataset data/law_extended.jsonl \
  --output adapters/law_r1.safetensors \
  --consent
```

### 本机私有训练

```bash
firefly-node train-local \
  --domain medical \
  --dataset data/medical_qa.jsonl \
  --output adapters/medical_local.safetensors
```

### 查看节点状态

```bash
firefly-node fed status
```

### 下载聚合权重

```bash
firefly-node fed download --round latest --output aggregated.safetensors
```

## 硬件要求

| GPU | VRAM | 可行性 |
|-----|------|--------|
| RTX 3060 | 12GB | ✅ QLoRA 4bit |
| RTX 4090 | 24GB | ✅ 推荐 |
| AutoDL 4090 | 24GB | ✅ 推荐 |
| Apple M系列 | 共享 | ⚠️ 实验性 |
| 无 GPU | — | ⚠️ 训练需 GPU，查询无需 |

## 数据格式

每行一个 JSON：
```json
{"instruction": "劳动合同试用期最长多久？", "output": "最长不超过六个月。"}
```

## 项目结构

```
firefly-client/
├── app/
│   ├── cli.py               # 命令行入口
│   ├── config.py            # 配置管理
│   ├── donation_bridge.py   # 信号回流
│   ├── executors/           # 执行器
│   │   └── fed_executor.py # 联邦执行器
│   ├── trainer/             # 训练器
│   │   └── real_trainer.py # QLoRA 训练内核
│   ├── task_executor.py     # 任务执行器
│   └── hardware.py          # 硬件检测
├── data/                    # 训练数据
├── scripts/                 # 工具脚本
│   └── fedavg.py           # FedAvg 聚合脚本
├── tests/                   # 单元测试
└── docs/                    # 基准测试
```

## 技术规格

- **底座模型**：Qwen2.5-1.5B-Instruct（4bit NF4 QLoRA）
- **LoRA 参数**：rank=8, alpha=16, targets=[q,v,k,o_proj + gate,up,down_proj]
- **训练框架**：transformers + peft + trl
- **聚合算法**：FedAvg 等权平均
- **调度中心**：http://106.14.220.169:8000

## 相关仓库

- [firefly-lm-org/scheduler](https://github.com/firefly-lm-org/scheduler) — 调度中心
- [firefly-lm-org/docs](https://github.com/firefly-lm-org/docs) — 文档与基准测试
- [firefly-lm-org/website](https://github.com/firefly-lm-org/website) — 官网

## License

OpenRAIL-M License
