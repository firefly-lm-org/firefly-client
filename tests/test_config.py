"""
tests/test_config.py · 配置模块测试
"""
import os
import pytest

from app.config import (
    ClientConfig,
    CONFIG_FILE,
    load_config,
    save_config,
    get_headers,
)


class TestLoadConfig:
    def test_load_config_default(self, isolated_home):
        """默认路径可读，文件不存在时返回默认值"""
        # 清理上测试遗留的 config 文件，确保干净状态
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()

        """默认路径可读，文件不存在时返回默认值"""
        cfg = load_config()
        assert isinstance(cfg, ClientConfig)
        assert cfg.server_url == "http://localhost:8000"
        assert cfg.access_token == ""
        assert cfg.username == ""
        assert cfg.node_id == ""

    def test_load_config_from_isolated_home(self, isolated_home):
        """FIREFLY_HOME 隔离目录中创建配置并读取"""
        cfg = ClientConfig(username="alice", server_url="http://test:9000")
        save_config(cfg)

        cfg2 = load_config()
        assert cfg2.username == "alice"
        assert cfg2.server_url == "http://test:9000"

    def test_save_and_load_roundtrip(self, isolated_home):
        """save_config → load_config 往返数据一致"""
        original = ClientConfig(
            server_url="http://example.com",
            access_token="tok123",
            refresh_token="ref456",
            user_id="uid789",
            username="bob",
            node_id="node001",
            node_name="bob-pc",
            status="running",
        )
        save_config(original)

        loaded = load_config()
        assert loaded.server_url == original.server_url
        assert loaded.access_token == original.access_token
        assert loaded.refresh_token == original.refresh_token
        assert loaded.user_id == original.user_id
        assert loaded.username == original.username
        assert loaded.node_id == original.node_id
        assert loaded.node_name == original.node_name
        assert loaded.status == original.status


class TestGetHeaders:
    def test_get_headers_without_token(self, isolated_home):
        """空 token 时 Authorization 不写入 header（代码逻辑：token 为空则不追加）"""
        cfg = ClientConfig()
        headers = get_headers(cfg)
        assert "Authorization" not in headers  # 空 token 不追加 Authorization
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_with_token(self, isolated_home):
        """有 token 时返回 Bearer header"""
        cfg = ClientConfig(access_token="mytoken123")
        headers = get_headers(cfg)
        assert headers["Authorization"] == "Bearer mytoken123"
        assert headers["Content-Type"] == "application/json"
