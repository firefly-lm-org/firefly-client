# ============================================================
# Firefly Client - GPU Training Container
# Tag: ghcr.io/firefly-lm-org/client:cu121-torch240-v1
# ============================================================
# Build & Push:
#   docker build -t ghcr.io/firefly-lm-org/client:cu121-torch240-v1 .
#   docker push ghcr.io/firefly-lm-org/client:cu121-torch240-v1
#
# Run (NVIDIA GPU):
#   docker run --gpus all ghcr.io/firefly-lm-org/client:cu121-torch240-v1 firefly start
#
# Run (with config mount):
#   docker run --gpus all -v $PWD/.firefly:/root/.firefly \
#     ghcr.io/firefly-lm-org/client:cu121-torch240-v1 \
#     firefly start --task-id <id>
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential zlib1g-dev libffi-dev \
    python3.11 python3-pip python3.11-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CUDA 12.1 FIRST (order matters for bitsandbytes)
RUN /opt/venv/bin/pip install \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies (exact versions from v0.3 validation)
RUN /opt/venv/bin/pip install \
    transformers==4.44.0 \
    peft==0.12.0 \
    trl==0.8.0 \
    bitsandbytes==0.50.0 \
    accelerate==1.14.0 \
    tokenizers==0.19.1 \
    huggingface-hub==0.36.2 \
    datasets>=2.16.0 \
    safetensors>=0.4.0 \
    numpy>=1.24.0

# Install CLI dependencies
RUN /opt/venv/bin/pip install \
    click>=8.1.0 \
    pydantic>=2.0.0 \
    requests>=2.31.0 \
    bcrypt==4.0.1 \
    python-dotenv>=1.0.0

# Create firefly user
RUN useradd -m -s /bin/bash firefly && \
    mkdir -p /home/firefly/.firefly && \
    chown -R firefly:firefly /home/firefly/.firefly

WORKDIR /home/firefly

# Copy client code (mount or copy)
COPY --chown=firefly:firefly . /home/firefly/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

USER firefly

ENTRYPOINT ["/opt/venv/bin/python3.11", "-m", "firefly"]
CMD ["--help"]
