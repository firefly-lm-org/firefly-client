"""
firefly-client · 硬件检测
自动识别 CPU / 内存 / GPU / 操作系统
"""
import platform
import os
import subprocess
import json


def detect_cpu_cores() -> int:
    """检测 CPU 核心数"""
    return os.cpu_count() or 1


def detect_memory_gb() -> float:
    """检测总内存（GB）"""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        elif system == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True,
            )
            bytes_ = int(result.stdout.strip())
            return round(bytes_ / 1024 / 1024 / 1024, 1)
        elif system == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            ctypes.c_longlong(kernel32.GetTickCount64())
            # 用 WMI 或 ctypes 获取内存
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                bytes_ = int(lines[1].strip())
                return round(bytes_ / 1024 / 1024 / 1024, 1)
    except Exception:
        pass
    return 8.0  # 默认值


def detect_gpu() -> tuple[str | None, float | None]:
    """
    检测 GPU 型号和显存（GB）
    优先尝试 torch.cuda，失败则尝试 nvidia-smi
    返回 (gpu_model, vram_gb)
    """
    # 方法 1：尝试 torch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = round(gpu_props.total_memory / 1024 / 1024 / 1024, 1)
            return gpu_name, vram_gb
    except ImportError:
        pass
    except Exception:
        pass

    # 方法 2：尝试 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(",")
                gpu_name = parts[0].strip()
                vram_mb = int(parts[1].strip())
                return gpu_name, round(vram_mb / 1024, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    return None, None


def detect_os() -> str:
    """检测操作系统"""
    system = platform.system()
    mapping = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}
    return mapping.get(system, "Unknown")


def full_hardware_report() -> dict:
    """返回完整硬件报告"""
    gpu_model, gpu_vram = detect_gpu()
    return {
        "cpu_cores": detect_cpu_cores(),
        "total_memory_gb": detect_memory_gb(),
        "gpu_model": gpu_model,
        "gpu_vram_gb": gpu_vram,
        "os_type": detect_os(),
    }


if __name__ == "__main__":
    report = full_hardware_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
