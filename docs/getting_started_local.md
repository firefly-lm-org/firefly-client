# 火种 Firefly LM —— 本地训练快速开始

> 适合：想用自己的数据微调 Qwen3-1.5B，但不想碰云端的开发者  
> 需要：Python 3.10+，NVIDIA GPU ≥ 8GB（或 AutoDL 4090，约 ¥1.5/h）  
> 预计耗时：约 40 分钟  
> 特点：**数据不出本机，无网络请求**

---

## 0 · 三种安装方式（选一种）

| 方案 | 命令 | 适用场景 |
|------|------|---------|
| **A 浅克隆（推荐）** | `git clone --depth 1 git@github.com:firefly-lm-org/firefly-client.git` | 正常网络，只取最新代码 |
| **B 国内镜像** | `git clone https://mirrors.tuna.tsinghua.edu.cn/git/firefly-lm-org/firefly-client.git` | GitHub 访问慢时 |
| **C PyPI（v0.7 后）** | `pip install firefly-node` | 一行安装，无需 clone |

### 方案 A 完整步骤（推荐）

```bash
# 1. 克隆
git clone --depth 1 git@github.com:firefly-lm-org/firefly-client.git
cd firefly-client

# 2. 安装（含 CLI 入口）
pip install -e .

# 3. 安装 GPU 依赖（国内镜像加速）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets sentencepiece \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装 unsloth（GPU 加速框架，必装）
pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **没有 GPU？** 去 [AutoDL](https://www.autodl.com) 租一张 RTX 4090（¥1.5/h），选 PyTorch 2.x 镜像，执行上面同样的命令。

### 验证安装

```bash
firefly-node --version
python -c "import torch; print('GPU:', torch.cuda.is_available()); print('VRAM:', f'{torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU only')"
```

---

## 1 · 准备数据（10 分钟）

需要一个 `.jsonl` 文件，每行一条问答：

```json
{"instruction": "劳动合同到期不续签有赔偿吗？", "output": "有赔偿。按《劳动合同法》第46条..."}
```

**直接用示例数据（推荐先跑通）：**

```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/law_qa.jsonl" -OutFile "data/law_qa.jsonl"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/education_qa.jsonl" -OutFile "data/education_qa.jsonl"

# Linux/Mac
curl -LO https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/law_qa.jsonl
curl -LO https://raw.githubusercontent.com/firefly-lm-org/firefly-client/main/data/education_qa.jsonl
```

**用自己的数据：** 准备 ≥20 条，格式同上，保存为 `my_data.jsonl`。

### 格式说明

| 字段 | 说明 |
|------|------|
| `instruction` | 用户的问题/指令 |
| `output` | 期望的回答 |

示例（法律领域）：
```json
{"instruction": "试用期最长多久？", "output": "试用期最长不超过六个月..."}
```

示例（教育领域）：
```json
{"instruction": "What is the past tense of 'go'?", "output": "The past tense of 'go' is 'went'..."}
```

---

## 2 · 训练（30 分钟）

```bash
firefly-node train-local \
    --dataset data/law_qa.jsonl \
    --domain law \
    --output my_law_adapter.safetensors \
    --steps 30
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` / `-d` | 必填 | JSONL 数据文件路径 |
| `--domain` | `general` | 领域标签（law/medical/python/tax/education，仅用于分类统计）|
| `--output` / `-o` | `firefly_adapter.safetensors` | 输出 LoRA 适配器文件路径 |
| `--steps` / `-s` | `30` | 训练步数（30 步约 30 分钟，效果够用；60-100 步效果更好）|
| `--base-model` | `unsloth/Qwen3-1.5B-Instruct-4bit` | 基础模型（需 HuggingFace 可访问）|

**训练过程示例：**

```
📦 领域: law | 数据: data/law_qa.jsonl | 步数: 30 | 基础模型: unsloth/Qwen3-1.5B-Instruct-4bit
  Step 10/30 | Loss: 0.521 | Elapsed: 120s
  Step 20/30 | Loss: 0.438 | Elapsed: 240s
  Step 30/30 | Loss: 0.382 | Elapsed: 360s

✅ 训练完成！
  文件: C:\...\my_law_adapter.safetensors (35.3 MB)
  最终 Loss: 0.382
  耗时: 362s
  可用 `firefly-node chat --adapter my_law_adapter.safetensors` 验证
```

---

## 3 · 推理验证（2 分钟）

**单次提问：**

```bash
firefly-node chat \
    --adapter my_law_adapter.safetensors \
    --prompt "劳动合同到期不续签有赔偿吗？"
```

**交互模式（多次问答）：**

```bash
firefly-node chat --adapter my_law_adapter.safetensors
```

```
💬 交互模式（输入空行退出）

你: 劳动合同到期不续签有赔偿吗？
模型: 有赔偿。按《劳动合同法》第46条，经济性裁员或合同期满不续订，
     用人单位应向劳动者支付经济补偿。每满一年支付一个月工资...

你: 加班费怎么算？
模型: 工作日加班支付1.5倍工资，休息日加班又不能安排补休的支付2倍
     工资，法定节假日加班支付3倍工资。依据《劳动法》第44条...
```

---

## 4 · 接下来可以做什么

| 目标 | 做法 |
|------|------|
| 换领域 | 准备 `medical_qa.jsonl`，`--domain medical` |
| 教育领域 | `--domain education`（已内置 20 条中英双语示例数据）|
| 效果更好 | 加数据到 50+ 条，加 `--steps 60` |
| 换小底座（省显存）| `--base-model unsloth/Qwen3-0.5B-Instruct-4bit`（RTX 3060 8GB 可跑）|
| 多领域聚合 | 见 `docs/getting_started_federated.md`（v0.7 联邦模式）|

---

## 常见问题

**Q: 数据会上传吗？**  
A: 不会。全程本地 GPU 训练，无任何网络请求。

**Q: 我的显卡够吗？**  
A: RTX 3060 12GB ✅ 可跑 1.5B 模型；4060/4070/4090 更快；8GB 卡用 `--base-model unsloth/Qwen3-0.5B-Instruct-4bit`。

**Q: 训练完能导出吗？**  
A: `my_law_adapter.safetensors` 是标准 PEFT LoRA 格式，可加载到任何兼容框架（transformers / vLLM / Ollama 等）。

**Q: 想要多领域知识融合怎么办？**  
A: 关注 `docs/getting_started_federated.md`，v0.7 联邦模式上线后可以参与多节点聚合训练。

**Q: 遇到 unsloth 安装失败？**  
A: 确保 CUDA 版本匹配：`pip install torch --index-url https://download.pytorch.org/whl/cu121`，然后重试 unsloth。

**Q: 遇到 Out of Memory？**  
A: 减小 max_seq_length：`FIREFLY_MAX_SEQ_LENGTH=256 firefly-node train-local ...` 或换小底座。
