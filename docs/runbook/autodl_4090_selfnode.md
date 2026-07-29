# AutoDL 4090 自测节点操作手册

## 适用场景
- 在 AutoDL 租一台 RTX 4090，同机运行调度中心 + 客户端
- 跑通第一次真实 QLoRA 训练，产出 `firefly_trainer_meta.json`
- 验证 v0.2 全链路

## 前置条件
- AutoDL 账号已注册、实名、充值 ≥¥50
- 本机已推送最新代码到 GitHub

## 操作步骤

### 1. 租用实例
- 地域：华东-杭州
- GPU：RTX 4090（24GB）
- 镜像：`PyTorch 2.5.1 / CUDA 12.4 / Python 3.12 (ubuntu22.04)`
- 数据盘：100GB
- 开机后复制 SSH 命令

### 2. SSH 登录
```bash
ssh -p <port> root@<host>
```

### 3. 拉代码
```bash
git clone https://github.com/firefly-lm-org/firefly-client.git
git clone https://github.com/firefly-lm-org/firefly-scheduler.git
cd firefly-client
```

### 4. 建 venv + 装 CUDA torch
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 5. 装其余依赖（阿里云镜像）
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  "unsloth[cu121]==2024.11.0" trl==0.9.6 \
  transformers==4.46.0 peft==0.13.0 accelerate==0.34.0 \
  safetensors==0.4.0 datasets==2.21.0 \
  requests redis pydantic click pytest
```

### 6. 验证 CUDA
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True NVIDIA GeForce RTX 4090
```

### 7. 同机起调度中心（终端 1）
```bash
cd ~/firefly-scheduler
source ../firefly-client/.venv/bin/activate
python -m app.main &
sleep 3
curl http://localhost:8000/health
# 期望: {"status":"ok"}
```

### 8. 跑真实训练（终端 2）
```bash
cd ~/firefly-client
source .venv/bin/activate
cp .env.example .env
export FIREFLY_MODEL_PATH=unsloth/Qwen3-1.5B-Instruct-4bit
export FIREFLY_LORA_RANK=8
export FIREFLY_MAX_SEQ_LENGTH=512
export FIREFLY_BATCH_SIZE=1
export FIREFLY_MAX_STEPS=60
export FIREFLY_DATA_PATH=data/alpaca_demo.jsonl
firefly-node start
```

### 9. 查看训练产物
```bash
cat ~/.firefly/train_output/firefly_trainer_meta.json
```

### 10. 拉回本机
在 AutoDL 终端复制 meta.json 内容，本机存为：
`docs/benchmarks/autodl-4090-qwen1.5b-r8.md`

### 11. 关机
AutoDL 控制台 → 关机（数据盘保留，按分钟停计费）

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `torch.cuda.is_available()` = False | 镜像没装 CUDA 驱动 | 换 `PyTorch 2.5.1 + CUDA 12.4` 镜像 |
| `pip install unsloth` 报错 | 缺 triton 或 CUDA 版本不匹配 | 用 `unsloth[cu121]` 而非 `unsloth` |
| 下载模型慢 | 境外 HF 限速 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| OOM | 参数太大 | 降 `MAX_SEQ_LENGTH` 到 256 或 `LORA_R` 到 4 |
