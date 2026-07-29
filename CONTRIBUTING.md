# 贡献指南

感谢你对火种 Firefly LM 项目的关注！以下是如何参与贡献的指南。

## 我能贡献什么

| 类型 | 说明 | 难度 |
|------|------|------|
| **数据** | 准备领域问答数据（JSONL 格式，≥20 条） | 低 |
| **训练** | 用你的 GPU 跑一轮训练，回传脱敏信号 | 低 |
| **测试** | 按文档跑通流程，报告问题 | 低 |
| **代码** | 修复 Bug、优化体验、实现新功能 | 中 |
| **文档** | 改进文档、翻译、写教程 | 低 |
| **推广** | 写使用体验帖、分享给朋友 | 低 |

## 快速开始

```bash
# 1. Fork 仓库
# 在 GitHub 上点击 Fork

# 2. 克隆你的 Fork
git clone --depth 1 https://github.com/<你的用户名>/firefly-client.git
cd firefly-client

# 3. 安装
pip install -e .
pip install torch transformers peft accelerate datasets sentencepiece

# 4. 跑通本地训练
firefly-node train-local --dataset data/law_qa.jsonl --steps 30

# 5. 联邦训练流程
firefly-node fed status              # 查看调度中心状态
firefly-node fed claim --domain law   # 认领任务
firefly-node fed train --task-id <id> --dataset data/law_qa.jsonl --steps 30
firefly-node fed complete --task-id <id> --loss 0.38 --samples 28
firefly-node fed download --round 1   # 下载聚合权重

# 6. 创建分支
git checkout -b fix/my-contribution

# 7. 修改代码

# 8. 提交
git add -A
git commit -m "fix: 简短描述你的修改"
git push origin fix/my-contribution

# 9. 创建 Pull Request
```

## 提交规范

```
<type>(<scope>): <description>

type: feat | fix | docs | style | refactor | test | chore
scope: cli | scheduler | trainer | docs | data
```

示例：
- `feat(cli): add fed claim command with mock fallback`
- `fix(scheduler): add public aggregation download endpoint`
- `docs: add rent_card_checklist.md`

## 信誉分规则

| 行为 | 信誉分变化 |
|------|-----------|
| 完成 1 个 task | +5 分 |
| 优秀完成（loss 很低） | +7 分 |
| 任务超时 | -10 分 |
| 虚假数据 | -50 分 |
| 连续失败 ≥3 次 | -5 分 |
| 最低分 | 0 分（封禁） |

信誉分 ≥30 可接高级任务，信誉分 0 永久封禁。

## 数据贡献

如果你想贡献领域数据（法律、医疗、教育等）：

1. 准备 JSONL 文件，每行格式：`{"instruction": "...", "output": "..."}`
2. 至少 20 条，确保内容准确、无版权问题
3. 提交到 `data/` 目录
4. PR 标题：`data: add <domain> dataset (<count> items)`

## 签署 CLA

提交 PR 前，请阅读并同意 [CLA.md](CLA.md)（贡献者许可协议）。在 PR 描述中添加：

```
I have read and agree to the CLA.
```

## 代码风格

- Python 3.10+
- 类型标注（typing）
- 中文注释和 docstring
- 每个函数有 docstring

## 测试

```bash
# 运行现有测试
python -m pytest tests/ -v
```

## 行为准则

- 尊重所有贡献者
- 数据隐私第一：不提交任何个人数据
- 诚实标注功能状态：已实现 / 预览 / 计划中

## 问题

有问题随时在 [GitHub Issues](https://github.com/firefly-lm-org/firefly-client/issues) 提问。
