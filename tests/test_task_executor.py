"""
tests/test_task_executor.py · 任务执行模块测试
使用 respx + temp file 模拟下载和上传流程
"""
import os
import io
import zipfile
import tempfile
import pytest
import respx
from httpx import Response

from app.config import ClientConfig
from app.task_executor import (
    download_task_package,
    upload_result,
    calc_sha256,
    make_result_package,
)


def make_zip_stream() -> bytes:
    """生成一个合法的内存 zip 文件流"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.bin", b"\x00" * 512)
    buf.seek(0)
    return buf.read()


class TestDownloadTaskPackage:
    @pytest.mark.asyncio
    async def test_download_creates_local_file(self, isolated_home, tmp_path):
        """mock 返回 200 + zip stream，验证本地文件被创建"""
        from app.task_executor import TASK_DIR
        import app.task_executor as te

        # 把 TASK_DIR 临时指向 tmp_path
        original = te.TASK_DIR
        te.TASK_DIR = tmp_path

        task_id = "task0001"
        zip_data = make_zip_stream()
        fake_url = "http://localhost:8000/download/task0001"

        try:
            with respx.mock:
                respx.get(fake_url).mock(
                    return_value=Response(
                        200,
                        content=zip_data,
                        headers={"content-length": str(len(zip_data))},
                    )
                )
                file_path = await download_task_package(fake_url, task_id, ClientConfig())

            assert file_path.exists()
            assert file_path.suffix == ".zip"
            assert file_path.stat().st_size > 0
        finally:
            te.TASK_DIR = original


class TestUploadResult:
    @pytest.mark.asyncio
    async def test_submit_result_success(self, isolated_home, tmp_path):
        """上传结果返回 200，不抛异常"""
        # 先生成一个结果 zip 文件
        result_zip = tmp_path / "result.zip"
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.bin", b"fake weights")

        cfg = ClientConfig(access_token="valid_tk")

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/task/submit-file").mock(
                return_value=Response(200, json={"task_id": "task0001", "status": "submitted"})
            )
            result = await upload_result(str(result_zip), "task0001", cfg)

        assert result["task_id"] == "task0001"
        assert result["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_submit_result_http_error(self, isolated_home, tmp_path):
        """上传返回非 200，抛出 RuntimeError"""
        result_zip = tmp_path / "result.zip"
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.bin", b"data")

        cfg = ClientConfig(access_token="valid_tk")

        with respx.mock:
            respx.post(f"{cfg.server_url}/api/v1/task/submit-file").mock(
                return_value=Response(500, json={"detail": "upload failed"})
            )
            with pytest.raises(RuntimeError, match="Upload failed"):
                await upload_result(str(result_zip), "task0001", cfg)


class TestMakeResultPackage:
    def test_make_result_package_returns_zip_path(self, isolated_home, tmp_path):
        """生成结果包返回 zip 文件路径，文件存在且可被打开"""
        task_id = "task0001"
        zip_path_str = make_result_package(task_id, tmp_path)

        assert os.path.exists(zip_path_str)
        assert zip_path_str.endswith(".zip")
        # 能正常打开
        with zipfile.ZipFile(zip_path_str, "r") as zf:
            names = zf.namelist()
            assert len(names) > 0


class TestCalcSha256:
    def test_calc_sha256_returns_hex_string(self, tmp_path):
        """calc_sha256 返回 64 位十六进制字符串"""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")

        sha = calc_sha256(str(f))
        assert isinstance(sha, str)
        assert len(sha) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in sha)

    def test_calc_sha256_deterministic(self, tmp_path):
        """同一文件多次计算结果一致"""
        f = tmp_path / "test.bin"
        f.write_bytes(b"test content")

        sha1 = calc_sha256(str(f))
        sha2 = calc_sha256(str(f))
        assert sha1 == sha2
