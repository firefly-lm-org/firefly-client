# Changelog

All notable changes to firefly-client are documented here.

## v0.6.0 (2026-07-29)

### Added
- 信誉分系统接入（调度中心侧）：查询/更新/历史/排行榜
- 信号回流协议（调度中心+客户端）：幂等写入/贡献积分
- FedExecutor（客户端侧）：认领/训练/进度/完成/下载 完整生命周期
- 冒烟测试脚本：test_v06_smoke.py（调度中心侧）
- 一键补训脚本：scripts/rent_and_run.sh（AutoDL 使用）
- GPU 运行清单：GPU_RUN_CHECKLIST.md
- 本地 FedAvg 聚合脚本：scripts/fedavg.py
- 领域数据集：law/medical/python/tax 各 30 条 QA

### Fixed
- PAT 泄露：git history 重写，GitHub Secret Scanning 已解除
- HTTPS 443 阻断：改 SSH 推送（端口 22）
- ED25519 SSH key 被 GitHub 拒绝：改 RSA 4096
- `federation_client.py` 参数名错误（--scheduler-url vs --server-url）
- 节点注册字段补全（reputation_score, signal_score）

### Changed
- 调度中心路由从 5 条扩展到 17 条
- 训练数据从 4 域扩展到 21 域
- 联邦训练闭环 Round 1 成功（4 节点，推理 4/4 正确）

## v0.5.0 (2026-07-26)

### Added
- 4 节点 FedAvg 聚合（等权 + 加权）
- 联邦训练 Round 1 完整闭环
- FedAvg 推理验证（法律/医疗/Python/财税 4 题全对）

### Fixed
- target_modules 必须统一为 7 个（解决 tensor key mismatch）
- trl 0.8.0 兼容（SFTConfig → TrainingArguments）
- firefly_trainer_meta.json 0KB bug

## v0.4.0 (2026-07-25)

### Added
- 本机 mock 联邦训练闭环验证
- 不等权 FedAvg 聚合脚本

## v0.3.0 (2026-07-25)

### Added
- 2 节点 FedAvg 聚合（Law + Medical）
- 推理验证（法律问题 + 医疗问题）
- AutoDL RTX 4090 真实 QLoRA 训练（60 步 / 51 秒）

## v0.2.0 (2026-07-24)

### Added
- firefly-client CLI 骨架
- 7 条命令：fed/start/stop/status, train-local, train-personal, train-npc, serve
- 硬件自动检测
- 断点续训逻辑骨架

## v0.1.0 (2026-07-23)

### Added
- 项目初始化
- 基础 CLI 框架
- 认证模块
