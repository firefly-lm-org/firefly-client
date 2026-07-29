# 常见问题 (FAQ)

## 环境相关

### Q: 我没有 GPU 能跑吗？

可以。demo mode 支持 CPU 模式，30 秒跑完。真实训练（`train-local`）需要 GPU，CPU 下约 2 小时跑 30 步。

### Q: 需要什么 Python 版本？

Python 3.8+。推荐 3.10+。运行 `firefly-node doctor` 检查环境。

### Q: torch 安装失败怎么办？

CPU 版：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU 版（CUDA 12.1）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q: safetensors 安装失败？

```bash
pip install safetensors
```
如果 PyPI 太慢，用国内镜像：
```bash
pip install safetensors -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练相关

### Q: 训练要多久？

| 硬件 | 30 步训练 | demo mode |
|------|----------|-----------|
| RTX 4090 | ~15 分钟 | <1 秒 |
| RTX 3060 | ~30 分钟 | <1 秒 |
| CPU | ~2 小时 | <1 秒 |

### Q: 数据格式是什么样的？

JSONL，每行一个 JSON 对象：
```json
{"instruction": "问题或指令", "output": "期望的回答"}
```

### Q: 训练完的适配器怎么用？

```bash
firefly-node chat --adapter my_adapter.safetensors
```

### Q: loss 不降怎么办？

1. 检查数据格式（instruction 和 output 不为空）
2. 确认 `--steps` 足够（至少 10 步）
3. 如果 loss 升高，可能是数据质量太差或学习率过高

### Q: 需要多少条数据？

| 用途 | 最少 | 推荐 |
|------|------|------|
| demo 验证 | 5 条 | 20 条 |
| 能用 | 20 条 | 50+ 条 |
| 生产 | 500+ 条 | 5000+ 条 |

## 联邦相关

### Q: 联邦学习和普通微调有什么区别？

普通微调：所有数据集中在一起训练。
联邦微调：每个节点用自己的数据本地训练，只上传模型权重，原始数据不离开本机。

### Q: 什么是 FedAvg？

FedAvg = Federated Averaging。把多个节点的模型权重按比例加权平均，得到一个联邦模型。Firefly 用软加权版（根据样本数和 holdout 提升度计算权重）。

### Q: 防作弊怎么工作？

Firefly 有两层防作弊：
- **L1 本地检测**：训练前对比 adapter 之间的 cos 相似度，低于阈值标记可疑
- **L1 服务器检测**：检查 loss 合理性、文件大小、提交频率、SHA256 记录

### Q: 怎么参与联邦？

```bash
# 1. 注册
firefly-node login

# 2. 认领任务
firefly-node fed-claim --domain law

# 3. 本地训练
firefly-node train-local --dataset my_data.jsonl --domain law --output adapter.safetensors

# 4. 提交权重
firefly-node fed-complete --task-id <task_id> --adapter adapter.safetensors

# 5. 聚合（由调度中心自动执行）
firefly-node fed-status
```

## 部署相关

### Q: 调度中心在哪？

阿里云服务器 `106.14.220.169:8000`，24 小时在线。

### Q: 代码在哪？

GitHub: https://github.com/firefly-lm-org/firefly-client

### Q: 怎么贡献代码？

1. Fork 仓库
2. 创建分支 `git checkout -b my-feature`
3. 提交 PR

### Q: 怎么报告 bug？

GitHub Issues: https://github.com/firefly-lm-org/firefly-client/issues
