# Firefly Client - One-click install for Windows
# Usage: .\install.ps1 [-GPU] [-Mirror]
#
#   -GPU    Install GPU training dependencies (requires NVIDIA GPU)
#   -Mirror Use Tsinghua mirror for non-CUDA packages (China users)
#
# Requirements:
#   - Python 3.9+ (download from python.org)
#   - For GPU: NVIDIA Driver >= 525 (for CUDA 12.1)
#   - Run as Administrator for system-wide install

param(
    [switch]$GPU,
    [switch]$Mirror
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Firefly Client Installer (Windows)" -ForegroundColor Cyan
Write-Host "  GPU mode: $GPU | Mirror: $Mirror" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$GPU_MODE = $GPU.IsPresent
$MIRROR   = $Mirror.IsPresent

# Check Python
Write-Host "[info] Checking Python..."
try {
    $pyVersion = python --version 2>&1
    Write-Host "  $pyVersion"
} catch {
    Write-Host "ERROR: Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check pip
try {
    python -m pip --version | Out-Null
} catch {
    Write-Host "ERROR: pip not found. Reinstall Python and enable 'Add to PATH'" -ForegroundColor Red
    exit 1
}

if ($GPU_MODE) {
    # GPU install
    Write-Host "[1/4] Checking NVIDIA GPU..." -ForegroundColor Yellow

    try {
        $gpuName = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        $driverVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
        Write-Host "  GPU: $gpuName" -ForegroundColor Green
        Write-Host "  Driver: $driverVersion" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: No NVIDIA GPU detected. Install NVIDIA Driver >= 525" -ForegroundColor Red
        Write-Host "Download: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Red
        exit 1
    }

    # Check torch
    $torchOk = python -c "import torch; assert torch.cuda.is_available()" 2>$null
    if (-not $torchOk) {
        Write-Host "[2/4] Installing PyTorch CUDA 12.1 (may take a few minutes)..." -ForegroundColor Yellow
        python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "[2/4] PyTorch CUDA detected — skipping" -ForegroundColor Green
    }

    Write-Host "[3/4] Installing GPU dependencies..." -ForegroundColor Yellow
    $pipArgs = @("-r", "requirements-gpu-rest.txt")
    if ($MIRROR) { $pipArgs += @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple") }
    python -m pip install @pipArgs

    Write-Host "[4/4] Verifying GPU..." -ForegroundColor Yellow
    python -c "
import torch
print('  GPU:', torch.cuda.get_device_name(0))
print('  CUDA:', torch.version.cuda)
print('  PASS: GPU training ready')
assert torch.cuda.is_available(), 'CUDA unavailable'
" 2>&1 | ForEach-Object { Write-Host "  $_" }

} else {
    # Mock install
    Write-Host "[1/3] Installing mock/CPU dependencies..." -ForegroundColor Yellow
    $pipArgs = @("-r", "requirements-mock.txt")
    if ($MIRROR) { $pipArgs += @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple") }
    python -m pip install @pipArgs

    Write-Host "[2/3] Installing firefly-client package..." -ForegroundColor Yellow
    python -m pip install -e .

    Write-Host "[3/3] Verification..." -ForegroundColor Yellow
    python -m firefly hardware 2>&1 | Select-Object -First 5
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Quick start:"
Write-Host "  firefly hardware         Check GPU status"
Write-Host "  firefly register         Register this node"
Write-Host "  firefly start            Start training"
Write-Host ""
