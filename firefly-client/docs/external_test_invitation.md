# 火种 Firefly LM —— 外部测试邀请

> 你的显卡 + 你的数据 = 你的微调模型。萤火虫只递扳手。

---

## 这是什么

火种（Firefly LM）是一个开源的 LLM 微调工具，让你在自己的显卡上微调 Qwen3-1.5B，数据不出本机。

- 官网：http://106.14.220.169/
- GitHub：https://github.com/firefly-lm-org/firefly-client
- 定位：300 块月费维持的调度中心，协调全球志愿算力做持续垂直微调

---

## 你需要

- NVIDIA GPU（≥ 8GB 显存），或去 [AutoDL](https://www.autodl.com) 租 RTX 4090（¥1.5/h）
- Python 3.10+
- 约 40 分钟

---

## 快速开始（5 步）

```bash
# 1. 克隆仓库
git clone --depth 1 https://github.com/firefly-lm-org/firefly-client.git
cd firefly-client

# 2. 安装 CLI
pip install -e .

# 3. 安装 GPU 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets sentencepiece unsloth \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 下载示例数据并训练（约 5 分钟）
curl -LO https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/law_qa.jsonl
firefly-node train-local --dataset law_qa.jsonl --domain law \
    --output my_adapter.safetensors --steps 30

# 5. 推理验证
firefly-node chat --adapter my_adapter.safetensors \
    --prompt "劳动合同到期不续签有赔偿吗？"
```

---

## 想试试联邦训练？（可选）

如果你愿意参与联邦微调（贡献脱敏信号，不传原始数据）：

```bash
# 查看调度中心状态
firefly-node fed status

# 认领任务（连不上调度中心时自动降级到本地模式）
firefly-node fed claim --domain law

# 本地训练
firefly-node fed train --dataset law_qa.jsonl --steps 30

# 回传脱敏信号（只传 task_id + loss + samples，不传数据）
firefly-node fed complete --task-id <your_task_id> --loss 0.38 --samples 28

# 下载聚合权重
firefly-node fed download --round latest
```

---

## 常见问题

**Q: 没有 GPU 怎么办？**
A: 去 AutoDL 租一张 RTX 4090，约 ¥1.5/h。选 PyTorch 2.x 镜像，跑上面同样的命令。详细步骤见 `docs/rent_card_checklist.md`。

**Q: 训练报错怎么办？**
A: 先跑 `firefly-node --help` 确认安装成功。如果 `import torch` 失败，检查 CUDA 版本是否匹配。

**Q: 我的数据格式是什么？**
A: JSONL 格式，每行一条 `{"instruction": "问题", "output": "答案"}`。详见 `docs/getting_started_local.md`。

**Q: 联邦训练会上传我的数据吗？**
A: 不会。只上传脱敏信号（task_id + loss + 样本数），原始数据始终在你的机器上。

---

## 反馈方式

- GitHub Issues：https://github.com/firefly-lm-org/firefly-client/issues
- 直接回复这条消息

遇到任何问题（报错、文档不清楚、命令不存在）请直接反馈，我们会快速修复。

---

*萤火虫和大厂模型是互补关系：大厂提供通用底座，萤火虫做持续垂直微调。代码免费，服务收费；本机训练，权重归你。*
