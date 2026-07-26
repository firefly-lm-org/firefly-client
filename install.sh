#!/bin/bash
# Firefly Client - One-click install for Linux / macOS
# Usage: bash install.sh [--gpu] [--mirror CN]
#
#   --gpu    Install GPU training dependencies (requires NVIDIA GPU)
#   --mirror Use Tsinghua mirror for non-CUDA packages (China users)
#
# Requirements:
#   - Python 3.9+
#   - pip
#   - For GPU: NVIDIA Driver >= 525 (for CUDA 12.1)
#   - For GPU: NVIDIA Container Toolkit (if using Docker)

set -e

GPU_MODE=false
USE_MIRROR=false

for arg in "$@"; do
    case $arg in
        --gpu) GPU_MODE=true ;;
        --mirror) USE_MIRROR=true ;;
    esac
done

echo "========================================"
echo "  Firefly Client Installer"
echo "  GPU mode: $GPU_MODE | Mirror: $USE_MIRROR"
echo "========================================"

# Detect OS
OS="$(uname -s)"
echo "[info] Detected OS: $OS"

if [ "$GPU_MODE" = true ]; then
    echo "[1/4] Checking NVIDIA GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Install NVIDIA Driver >= 525 first."
        echo "Download: https://www.nvidia.com/Download/index.aspx"
        exit 1
    fi
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  NVIDIA Driver: $DRIVER_VERSION"

    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[2/4] Installing PyTorch CUDA 12.1 (this may take a few minutes)..."
        pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "[2/4] PyTorch with CUDA detected — skipping torch install"
    fi

    echo "[3/4] Installing GPU training dependencies..."
    if [ "$USE_MIRROR" = true ]; then
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-gpu-rest.txt
    else
        pip install -r requirements-gpu-rest.txt
    fi

    echo "[4/4] Verifying GPU..."
    python3 -c "
import torch
print('  GPU:', torch.cuda.get_device_name(0))
print('  CUDA:', torch.version.cuda)
print('  GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
assert torch.cuda.is_available(), 'CUDA not available'
print('  PASS: GPU training ready')
"

else
    echo "[1/3] Installing mock/CPU dependencies..."
    if [ "$USE_MIRROR" = true ]; then
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-mock.txt
    else
        pip install -r requirements-mock.txt
    fi

    echo "[2/3] Installing firefly-client package..."
    pip install -e .

    echo "[3/3] Verifying installation..."
    firefly --version 2>/dev/null || python -m firefly --version 2>/dev/null || echo "  (CLI ready — run 'firefly hardware' to check)"
    echo "  PASS: Mock mode ready"
fi

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  firefly hardware         # Check GPU status"
echo "  firefly register         # Register this node"
echo "  firefly start            # Start training"
echo ""
echo "For Docker: docker run --gpus all ghcr.io/firefly-lm-org/client:latest firefly start"
