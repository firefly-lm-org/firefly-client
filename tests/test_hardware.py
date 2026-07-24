"""
tests/test_hardware.py · 硬件检测模块测试
"""
import pytest

from app.hardware import (
    detect_cpu_cores,
    detect_memory_gb,
    detect_gpu,
    detect_os,
    full_hardware_report,
)


class TestDetectHardware:
    def test_detect_hardware_returns_dict(self):
        """返回 dict，含必需的键"""
        report = full_hardware_report()
        assert isinstance(report, dict)
        assert "cpu_cores" in report
        assert "total_memory_gb" in report
        assert "os_type" in report

    def test_cpu_cores_positive(self):
        """cpu_cores >= 1"""
        cores = detect_cpu_cores()
        assert isinstance(cores, int)
        assert cores >= 1

    def test_memory_gb_positive(self):
        """memory_gb > 0"""
        mem = detect_memory_gb()
        assert isinstance(mem, float)
        assert mem > 0

    def test_detect_os_returns_string(self):
        """detect_os 返回已知的操作系统名称"""
        os_type = detect_os()
        assert isinstance(os_type, str)
        assert os_type in ("Linux", "Darwin", "Windows", "Unknown")

    def test_detect_gpu_returns_tuple(self):
        """detect_gpu 返回 (model, vram) 元组"""
        gpu_model, gpu_vram = detect_gpu()
        assert gpu_model is None or isinstance(gpu_model, str)
        assert gpu_vram is None or isinstance(gpu_vram, float)
