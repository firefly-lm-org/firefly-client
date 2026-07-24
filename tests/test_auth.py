"""
tests/test_auth.py · 认证模块测试
使用 respx 拦截 httpx.AsyncClient 请求
"""
import pytest
import respx
from httpx import Response

from app.config import ClientConfig, load_config, save_config
from app.auth import register, login, refresh_token, ensure_authenticated


def make_token_response(access_token="tk", refresh_token="rk", user_id="uid", username="alice"):
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user_id": user_id,
        "username": username,
    }


class TestRegister:
    @pytest.mark.asyncio
    async def test_register_success(self, isolated_home):
        """注册成功返回 True，config.json 写入 token"""
        cfg = ClientConfig()

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/register").mock(
                return_value=Response(200, json=make_token_response())
            )
            result = await register(cfg, "alice", "pass123")

        assert result is True
        saved = load_config()
        assert saved.access_token == "tk"
        assert saved.refresh_token == "rk"
        assert saved.user_id == "uid"

    @pytest.mark.asyncio
    async def test_register_http_error(self, isolated_home):
        """HTTP 错误不抛异常，返回 False"""
        cfg = ClientConfig()

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/register").mock(
                return_value=Response(500, json={"detail": "server error"})
            )
            result = await register(cfg, "alice", "pass123")

        assert result is False


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_success(self, isolated_home):
        """登录成功返回 True，写入 config"""
        cfg = ClientConfig()

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/login").mock(
                return_value=Response(200, json=make_token_response(username="bob"))
            )
            result = await login(cfg, "bob", "secret")

        assert result is True
        saved = load_config()
        assert saved.username == "bob"
        assert saved.access_token == "tk"

    @pytest.mark.asyncio
    async def test_login_http_error(self, isolated_home):
        """登录失败返回 False"""
        cfg = ClientConfig()

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/login").mock(
                return_value=Response(401, json={"detail": "invalid credentials"})
            )
            result = await login(cfg, "bob", "wrong")

        assert result is False


class TestRefreshToken:
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, isolated_home):
        """刷新成功更新 access_token"""
        cfg = ClientConfig(refresh_token="old_refresh")
        save_config(cfg)

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/refresh").mock(
                return_value=Response(200, json={"access_token": "new_token"})
            )
            result = await refresh_token(cfg)

        assert result is True
        assert cfg.access_token == "new_token"

    @pytest.mark.asyncio
    async def test_refresh_token_no_refresh_token(self, isolated_home):
        """没有 refresh_token 时返回 False，不抛异常"""
        cfg = ClientConfig()
        result = await refresh_token(cfg)
        assert result is False


class TestEnsureAuthenticated:
    @pytest.mark.asyncio
    async def test_ensure_authenticated_with_valid_token(self, isolated_home):
        """有 access_token 时直接返回 True"""
        cfg = ClientConfig(access_token="valid_tk")
        result = await ensure_authenticated(cfg)
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_authenticated_without_token_but_has_refresh(self, isolated_home):
        """无 access_token 但有 refresh_token 时自动刷新"""
        cfg = ClientConfig(refresh_token="refresh_tk")
        save_config(cfg)

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/auth/refresh").mock(
                return_value=Response(200, json={"access_token": "new_tk"})
            )
            result = await ensure_authenticated(cfg)

        assert result is True
        assert cfg.access_token == "new_tk"

    @pytest.mark.asyncio
    async def test_ensure_authenticated_without_any_token(self, isolated_home):
        """既无 access_token 也无 refresh_token 时返回 False"""
        cfg = ClientConfig()
        result = await ensure_authenticated(cfg)
        assert result is False
