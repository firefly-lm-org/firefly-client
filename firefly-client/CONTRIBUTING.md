# 参与指南

## 谁可以参与

- 有 NVIDIA GPU（≥8GB VRAM）的开发者
- 愿意运行 `firefly-node fed start` 并回传 adapter sha256
- 不要求传原始数据，只传脱敏信号

## 如何开始

1. 阅读 [README.md](README.md) 安装客户端
2. 运行 `firefly-node fed status` 确认连上调度中心
3. 运行 `firefly-node fed start --domain law --consent`
4. 把输出的 task_id 和 sha256 贴在 GitHub Issues

## 贡献类型

| 类型 | 说明 |
|------|------|
| 跑训练 | 用自己显卡跑联邦任务，获得信誉分 |
| 提数据 | 提供脱敏后的领域数据 |
| 修代码 | PR 修 bug 或加功能 |
| 写文档 | 改进 README / 教程 |

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

## 本地开发

```bash
git clone git@github.com:firefly-lm-org/firefly-client.git
cd firefly-client
pip install -r requirements.txt
python -m app.main --help
pytest tests/
```

## 联系方式

- GitHub Issues：bug 报告 / 功能请求
- GitHub Discussions：讨论 / 问答
