"""
tests/conftest.py · pytest fixtures

CRITICAL execution order (pytest):
  1. pytest loads conftest.py
  2. conftest module-level code runs  ← we set FIREFLY_HOME HERE
  3. pytest collects test modules (imports happen here → app.* sees correct FIREFLY_HOME)
  4. pytest executes session/function fixtures
  5. tests run

Setting FIREFLY_HOME at module load time (step 2) guarantees that
app.config's CONFIG_DIR / CONFIG_FILE = the isolated temp dir.
"""
import os
import sys
import tempfile
import shutil

import pytest

# ── ISOLATION: 必须在这里（模块加载时）设置 FIREFLY_HOME ──
# 此时 app.config 尚未被任何 test 模块导入，路径求值将使用正确的隔离目录
_isolated_dir = tempfile.mkdtemp(prefix="firefly_test_")
_environ_backup = os.environ.get("FIREFLY_HOME")

os.environ["FIREFLY_HOME"] = _isolated_dir


def pytest_sessionfinish(session, exitstatus):
    """整个 session 结束后清理隔离目录"""
    shutil.rmtree(_isolated_dir, ignore_errors=True)
    if _environ_backup is not None:
        os.environ["FIREFLY_HOME"] = _environ_backup
    elif "FIREFLY_HOME" in os.environ:
        del os.environ["FIREFLY_HOME"]


@pytest.fixture
def isolated_home():
    """返回隔离目录路径"""
    return _isolated_dir
