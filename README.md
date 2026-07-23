# 🔥 Firefly Client · 火种客户端

> 安装了此客户端的闲置 GPU 机器，将成为萤火虫大模型的"火种节点"

## 安装

```bash
pip install -r requirements.txt
```

## 首次设置

```bash
python -m firefly.cli setup
# 按提示输入调度中心地址 + 你的账号密码
# 程序会自动检测 GPU 并注册节点
```

## 运行

```bash
# 持续运行，抢任务训练
python -m firefly.cli run

# 只领取一个任务然后退出（适合测试）
python -m firefly.cli run --once
```

## 工作原理

1. **心跳上报**：客户端每分钟向调度中心上报节点状态
2. **任务抢领**：调度中心有空闲任务时，客户端竞争领取
3. **下载训练数据**：从 MinIO 自动下载 JSONL 格式的训练语料
4. **QLoRA 训练**：4-bit量化 LoRA 微调，自动选择最优 batch size
5. **结果上报**：训练完成后上报贡献分，清理显存，等待下一轮

## 硬件要求

| 配置 | 说明 |
|------|------|
| GPU 显存 | ≥ 6GB（RTX 3060 及以上） |
| 内存 | ≥ 16GB |
| 存储 | ≥ 20GB（用于模型缓存） |
| 网络 | 上行 ≥ 5Mbps（上传权重） |

## 自动检测

运行 `python -m firefly.core.hardware` 可查看本机硬件信息。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FIREFLY_SCHEDULER_URL` | `http://localhost:8000` | 调度中心地址 |
| `AWS_ACCESS_KEY_ID` | `firefly_access` | MinIO 访问密钥 |
| `AWS_SECRET_ACCESS_KEY` | `firefly_secret` | MinIO 访问密码 |
| `FIREFLY_S3_ENDPOINT` | `http://localhost:9000` | S3 端点 |
| `FIREFLY_S3_BUCKET` | `firefly-models` | 存储桶名 |
