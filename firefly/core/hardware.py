"""硬件检测：GPU/CPU/RAM 信息采集"""
import platform
import subprocess
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HardwareInfo:
    gpu_model: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_count: int = 0
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None
    supports_bf16: bool = False
    max_batch_size: int = 1
    capabilities: list[str] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


def get_hardware_info() -> HardwareInfo:
    """采集本机硬件信息，跨平台"""
    info = HardwareInfo(
        cpu_cores=platform.os.cpu_count(),
        ram_gb=_get_ram_gb(),
    )

    if platform.system() == "Windows":
        _detect_nvidia_windows(info)
    elif platform.system() == "Linux":
        _detect_nvidia_linux(info)
    elif platform.system() == "Darwin":
        _detect_mps_mac(info)

    _estimate_batch_size(info)
    return info


def _get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0


def _detect_nvidia_windows(info: HardwareInfo):
    """Windows: 通过 nvidia-smi 检测 NVIDIA GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return
        lines = result.stdout.strip().split("\n")
        info.gpu_count = len(lines)
        if lines and lines[0]:
            parts = lines[0].split(", ")
            info.gpu_model = parts[0].strip()
            vram_mb = float(parts[1].strip())
            info.gpu_vram_gb = round(vram_mb / 1024, 1)
            if len(parts) >= 4:
                cc = parts[3].strip()  # e.g. "8.9"
                if float(cc) >= 8.0:
                    info.supports_bf16 = True
            info.capabilities.append("cuda")
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        pass


def _detect_nvidia_linux(info: HardwareInfo):
    """Linux: nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            info.gpu_count = len(lines)
            if lines and lines[0]:
                parts = lines[0].split(",")
                info.gpu_model = parts[0].strip()
                info.gpu_vram_gb = round(float(parts[1].strip()) / 1024, 1)
                info.supports_bf16 = True
                info.capabilities.append("cuda")
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        pass


def _detect_mps_mac(info: HardwareInfo):
    """macOS: Apple Silicon MPS"""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cpu = result.stdout.strip()
            info.gpu_model = f"Apple Silicon ({cpu})"
            info.gpu_count = 1
            info.capabilities.append("mps")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


def _estimate_batch_size(info: HardwareInfo):
    """根据显存估算最大 batch size（per_device_train_batch_size）"""
    if info.gpu_vram_gb is None or info.gpu_vram_gb <= 0:
        return
    if info.gpu_vram_gb >= 40:
        info.max_batch_size = 16
    elif info.gpu_vram_gb >= 24:
        info.max_batch_size = 8
    elif info.gpu_vram_gb >= 16:
        info.max_batch_size = 4
    elif info.gpu_vram_gb >= 12:
        info.max_batch_size = 2
    elif info.gpu_vram_gb >= 8:
        info.max_batch_size = 1
    else:
        info.max_batch_size = 1
        info.capabilities.append("low_vram")


if __name__ == "__main__":
    info = get_hardware_info()
    print(json.dumps(asdict(info), indent=2, ensure_ascii=False))
