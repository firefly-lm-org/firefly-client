"""
firefly-client · 配置管理
读取配置目录下的 config.json，管理 token / 服务器地址 / 节点信息
"""
import os
import json
from pathlib import Path
from pydantic import BaseModel


# ── 配置文件路径 ──────────────────────
# 允许通过环境变量 FIREFLY_HOME 指定配置目录（隔离/测试用）
CONFIG_DIR = Path(os.environ.get("FIREFLY_HOME", str(Path.home() / ".firefly")))
CONFIG_FILE = CONFIG_DIR / "config.json"


class ClientConfig(BaseModel):
    """客户端配置（持久化到 JSON）"""
    server_url: str = "http://localhost:8000"
    access_token: str = ""
    refresh_token: str = ""
    user_id: str = ""
    username: str = ""
    node_id: str = ""
    node_name: str = ""
    status: str = "stopped"  # running / stopped / paused


def load_config() -> ClientConfig:
    """从磁盘加载配置，不存在或损坏则返回默认值"""
    if not CONFIG_FILE.exists():
        return ClientConfig()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return ClientConfig(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ClientConfig()


def save_config(cfg: ClientConfig):
    """保存配置到磁盘"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump()
    CONFIG_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_headers(cfg: ClientConfig) -> dict:
    """构造带 Authorization 的请求头"""
    headers = {"Content-Type": "application/json"}
    if cfg.access_token:
        headers["Authorization"] = f"Bearer {cfg.access_token}"
    return headers
