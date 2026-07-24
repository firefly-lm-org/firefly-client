"""
tests/test_node.py · 节点模块测试
使用 respx 拦截 httpx.AsyncClient 请求
"""
import pytest
import respx
from httpx import Response

from app.config import ClientConfig, load_config
from app.node import register_node, query_status


class TestRegisterNode:
    @pytest.mark.asyncio
    async def test_register_node_success(self, isolated_home):
        """节点注册成功，config 写入 node_id"""
        cfg = ClientConfig(
            access_token="valid_tk",
            user_id="user123",
            username="alice",
        )

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/node/register").mock(
                return_value=Response(200, json={
                    "node_id": "node_abc123def456",
                    "node_name": "alice-pc",
                    "status": "online",
                    "reputation_score": 100.0,
                    "max_task_level": 3,
                })
            )
            result = await register_node(cfg, "alice-pc")

        assert result is True
        saved = load_config()
        assert saved.node_id == "node_abc123def456"
        assert saved.node_name == "alice-pc"

    @pytest.mark.asyncio
    async def test_register_node_http_error(self, isolated_home):
        """注册请求返回错误码，不抛异常，返回 False"""
        cfg = ClientConfig(access_token="valid_tk")

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/node/register").mock(
                return_value=Response(500, json={"detail": "internal error"})
            )
            result = await register_node(cfg, "alice-pc")

        assert result is False


class TestQueryStatus:
    @pytest.mark.asyncio
    async def test_get_node_status_success(self, isolated_home):
        """查询节点状态成功返回 True"""
        cfg = ClientConfig(access_token="valid_tk", node_id="node_abc123")

        with respx.mock:
            respx.get(f"{cfg.server_url}/api/v1/node/status").mock(
                return_value=Response(200, json={
                    "node_id": "node_abc123def456",
                    "node_name": "alice-pc",
                    "status": "online",
                    "reputation_score": 100.0,
                    "max_task_level": 3,
                    "last_heartbeat": "2026-07-24T10:00:00Z",
                })
            )
            result = await query_status(cfg)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_node_status_http_error(self, isolated_home):
        """查询失败返回 False"""
        cfg = ClientConfig(access_token="valid_tk")

        with respx.mock:
            respx.get(f"{cfg.server_url}/api/v1/node/status").mock(
                return_value=Response(403, json={"detail": "forbidden"})
            )
            result = await query_status(cfg)

        assert result is False
