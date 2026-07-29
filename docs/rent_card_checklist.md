# 租卡清单（贴墙上，租卡时照着做）

> AutoDL 租 4090 一次跑通火种全流程的操作手册
> 预计花费：¥1.5/小时 × 2 小时 = ¥3

## 1. 租卡（5 分钟）

1. 打开 https://www.autodl.com 注册/登录
2. 选择「算力市场」→ 筛选 **4090 (24GB)**
3. 选择最便宜的（¥1.5-2.0/小时）
4. 镜像选择：**PyTorch 2.1.0 + Python 3.10 + CUDA 12.1**
5. 创建实例，等待启动（约 1 分钟）

## 2. 创建干净环境（3 分钟）

```bash
# SSH 进实例（AutoDL 会给你 SSH 命令）

# 创建独立环境
conda create -n firefly python=3.10 -y
conda activate firefly

# 安装训练依赖
pip install torch transformers peft accelerate datasets sentencepiece
pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 期望输出: CUDA: True, GPU: NVIDIA GeForce RTX 4090
```

## 3. 克隆代码（2 分钟）

```bash
git clone --depth 1 https://github.com/firefly-lm-org/firefly-client.git ~/firefly-client
cd ~/firefly-client
pip install -e .
```

## 4. 本地训练验证（30 分钟）

```bash
# 法律领域
firefly-node train-local \
  --dataset data/law_qa.jsonl \
  --steps 30 \
  --output-dir ~/firefly/train_output/law_r1

# 医疗领域
firefly-node train-local \
  --dataset data/medical_qa.jsonl \
  --steps 30 \
  --output-dir ~/firefly/train_output/medical_r1

# Python 领域
firefly-node train-local \
  --dataset data/python_qa.jsonl \
  --steps 30 \
  --output-dir ~/firefly/train_output/python_r1

# 税务领域
firefly-node train-local \
  --dataset data/tax_qa.jsonl \
  --steps 30 \
  --output-dir ~/firefly/train_output/tax_r1
```

## 5. 联邦训练验证（15 分钟）

```bash
# 查看调度中心状态
firefly-node fed status

# 认领任务
firefly-node fed claim --domain law

# 联邦训练
firefly-node fed train \
  --dataset data/law_qa.jsonl \
  --output fed_law_r1.safetensors \
  --steps 30

# 回传脱敏信号
firefly-node fed complete \
  --task-id law_r1_001 \
  --loss 0.38 \
  --samples 28

# 下载聚合权重
firefly-node fed download --round latest --output aggregated.safetensors
```

## 6. 推理验证（5 分钟）

```bash
# 用 Python 脚本推理
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('unsloth/Qwen3-1.5B-Instruct-4bit')
tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen3-1.5B-Instruct-4bit')
model = PeftModel.from_pretrained(base, '~/firefly/train_output/law_r1/lora.safetensors')

for q in ['劳动合同到期不续签有赔偿吗？', '加班费怎么算？', '试用期被辞退有赔偿吗？']:
    inputs = tokenizer(q, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(f'Q: {q}')
    print(f'A: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')
    print()
"
```

## 7. 打包结果（5 分钟）

```bash
# 打包训练权重
tar czf ~/firefly_results.tar.gz -C ~/firefly/train_output .

# 传回本机（用 AutoDL 的文件管理或 scp）
# AutoDL 文件管理：在网页端下载 ~/firefly_results.tar.gz
```

## 8. 关机省钱

```bash
# 训练完成后立即关机
# AutoDL 网页端 → 实例 → 关机
# 关机不收费，只有开机时才计费
```

## 费用估算

| 步骤 | 时间 | 费用 (¥1.5/h) |
|------|------|------|
| 租卡 + 环境 | 10 min | ¥0.25 |
| 本地训练 ×4 | 30 min | ¥0.75 |
| 联邦训练 | 15 min | ¥0.38 |
| 推理验证 | 5 min | ¥0.12 |
| 打包传输 | 5 min | ¥0.12 |
| **总计** | **~65 min** | **~¥1.62** |

## 常见问题

**Q: 租什么卡？**
A: 4090 (24GB) 最划算。3060 (12GB) 也能跑但慢 3 倍。A100 太贵不推荐。

**Q: 训练中断了怎么办？**
A: 权重按 task_id 分目录保存，重新跑同样命令即可。不会覆盖已有权重。

**Q: 怎么省钱？**
A: 用 `--steps 30` 而非 60；4 个领域串行跑；跑完立即关机。

**Q: unsloth 安装失败怎么办？**
A: 用清华镜像：`pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple`。或用 `--mock` 模式先验证流程。

**Q: 可以用 CPU 跑吗？**
A: 可以用 `--mock` 模式验证流程（不加载真实模型），但真实训练必须有 GPU。
