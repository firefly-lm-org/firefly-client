# 火种 Firefly LM —— 联邦训练（路线图 · v0.7 目标）

> ⚠️ **本文档描述计划中的联邦流程，当前（v0.6）部分功能未上线**  
> **现在可用的是本地训练版**：[`docs/getting_started_local.md`](./getting_started_local.md)

---

## 什么是联邦训练

多个参与者各自用自己的显卡和数据训练，**只上传 LoRA 权重**，原始数据永远不出本机。调度中心收集所有人的权重后做 FedAvg 聚合，生成一个融合了所有人知识的模型。

```
你的电脑（法律数据）
    ↓ LoRA 权重（几 MB）
调度中心（聚合）
    ↑ LoRA 权重（几 MB）
其他人的电脑（医疗/编程/税务数据）
    ↓
融合模型（所有人共享）
```

---

## 当前状态（v0.6）

| 组件 | 状态 | 说明 |
|------|------|------|
| 调度中心（阿里云） | ✅ 在线 | http://106.14.220.169:8000，17 条路由 |
| 信誉分系统 | ✅ 完成 | 100 分起步，完成任务 +5 |
| 信号回流协议 | ✅ 完成 | 幂等写入，脱敏信号 |
| FedExecutor 代码 | ✅ 完成 | 在 `app/executors/` 里 |
| **fed CLI 子命令** | 🔄 进行中 | `firefly-node fed` 命令实现中 |
| **任务池管理** | 🔄 进行中 | admin 端点创建任务池 |
| **多节点真实联调** | ⏳ 待做 | 至少 2 个节点联调通过 |

---

## 计划流程（v0.7 目标）

### 第一步：注册并登录

```bash
# 注册账户（只需一次）
firefly-node register --username alice --password yourpassword

# 登录
firefly-node login --username alice --password yourpassword
```

### 第二步：注册节点

```bash
# 将本机注册为算力节点
firefly-node node-register my-laptop
# 输出：✅ 节点注册成功，信誉分: 100
```

### 第三步：认领任务

```bash
# 从调度中心认领一个法律领域的训练任务
firefly-node fed claim --domain law

# 如果调度中心没有可用任务，自动切换到本地 Mock 模式（不影响训练）
# 输出：✅ 认领成功: law_r6_001 或 ⚠️ Mock 任务: mock_law_r1
```

### 第四步：训练并回传

```bash
# 用自己的数据训练（30 分钟）
firefly-node fed train \
    --dataset my_law_qa.jsonl \
    --output law_r6.safetensors \
    --task-id law_r6_001

# 训练完成后自动回传权重和脱敏信号
# 输出：✅ 训练完成 | Loss: 0.382 | 信誉分奖励: +5（当前 105）
```

> **关键：你的原始数据不会上传。只回传一个脱敏信号（任务ID + Loss + 样本数）。**

### 第五步：下载聚合权重

```bash
# 其他节点也在训练，下载最新聚合结果
firefly-node fed download --round latest --output aggregated.safetensors

# 输出：✅ 下载成功
# 聚合轮次: r5_full | 参与节点: 4 | SHA256: 653bde6c0740b8e5...
```

---

## 什么时候能用

跟踪 GitHub 里程碑 `docs/milestones.md` 的 v0.7 条目。

**预计完成条件：**
1. `firefly-node fed claim / train / complete / download` 全部实现
2. 调度中心有活跃任务池（至少 4 个 pending 任务）
3. 至少 2 个真实节点联调通过

**现在能做什么：**
- 用本地训练版（`getting_started_local.md`）先跑通你的数据
- 你的适配器格式完全兼容联邦模式，无需重新训练
- 准备好数据，等 v0.7 上线后直接参与联邦

---

## 参与贡献（帮助加速 v0.7）

| 方式 | 说明 |
|------|------|
| 贡献数据 | 提供脱敏后的领域问答数据（≥20 条）|
| 测试反馈 | 在 GitHub Issues 报告 `fed` 命令的 bug |
| 代码贡献 | 见 `CONTRIBUTING.md` |
| 算力贡献 | 有 GPU 的开发者可以参与多节点联调测试 |
