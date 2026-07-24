"""
tests/test_main_cli.py · CLI 主入口测试

关键：patch 路径必须对到 main.py 里的别名绑定名：
  from app.auth import register as auth_register, login as auth_login
→ patch "app.main.auth_register" 而非 "app.auth.register"
"""
import pytest
from unittest.mock import patch, AsyncMock
from click.testing import CliRunner
from typer.testing import _get_command

from app.main import app as typer_app

_clirunner = CliRunner()


def invoke(args):
    patched_app = _get_command(typer_app)
    return _clirunner.invoke(patched_app, args)


# mock 函数（同步，auth_register/auth_login 是同步包装器）
def mock_register_ok(cfg, username, password):
    return True


def mock_register_err(cfg, username, password):
    raise Exception("network error")


def mock_login_ok(cfg, username, password):
    return True


def mock_login_err(cfg, username, password):
    raise Exception("network error")


# ---------- tests ----------

class TestConfigShow:
    def test_config_show_command(self, isolated_home):
        result = invoke(["config-show"])
        assert result.exit_code == 0

    def test_config_show_help(self, isolated_home):
        result = invoke(["config-show", "--help"])
        assert result.exit_code == 0


class TestRegister:
    def test_register_help(self, isolated_home):
        result = invoke(["register", "--help"])
        assert result.exit_code == 0
        assert "--username" in result.output

    def test_register_with_password_calls_auth_ok(self, isolated_home):
        """--username + --password 完整参数 → auth_register 被调用"""
        with patch("app.main.auth_register", side_effect=mock_register_ok) as m:
            result = invoke(["register", "--username", "alice", "--password", "secret123"])
            assert result.exit_code == 0
            m.assert_called_once()
            _, username, password = m.call_args[0]
            assert username == "alice"
            assert password == "secret123"

    def test_register_network_error_shows_message(self, isolated_home):
        """网络错误 → exit_code != 0"""
        with patch("app.main.auth_register", side_effect=mock_register_err):
            result = invoke(["register", "--username", "alice", "--password", "secret123"])
            assert result.exit_code != 0


class TestLogin:
    def test_login_help(self, isolated_home):
        result = invoke(["login", "--help"])
        assert result.exit_code == 0

    def test_login_with_password_ok(self, isolated_home):
        with patch("app.main.auth_login", side_effect=mock_login_ok) as m:
            result = invoke(["login", "--username", "alice", "--password", "secret123"])
            assert result.exit_code == 0
            m.assert_called_once()
            _, username, password = m.call_args[0]
            assert username == "alice"
            assert password == "secret123"

    def test_login_network_error(self, isolated_home):
        with patch("app.main.auth_login", side_effect=mock_login_err):
            result = invoke(["login", "--username", "alice", "--password", "secret123"])
            assert result.exit_code != 0


class TestNodeRegister:
    def test_node_register_help(self, isolated_home):
        result = invoke(["node-register", "--help"])
        assert result.exit_code == 0

    def test_node_register_requires_login(self, isolated_home):
        """未登录（无 config）→ 失败"""
        result = invoke(["node-register", "--name", "mynode"])
        assert result.exit_code != 0


class TestStatus:
    def test_status_help(self, isolated_home):
        result = invoke(["status", "--help"])
        assert result.exit_code == 0


class TestStats:
    def test_stats_help(self, isolated_home):
        result = invoke(["stats", "--help"])
        assert result.exit_code == 0


class TestStart:
    def test_start_help(self, isolated_home):
        result = invoke(["start", "--help"])
        assert result.exit_code == 0
