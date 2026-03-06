# ComfyUI with SageAttention for NVIDIA Blackwell (RTX PRO 6000, RTX 50 series)
# Using PyTorch 2.7.1 + CUDA 12.8 (proven to work with Blackwell)

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Install dependencies required for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Triton (required for SageAttention)
RUN pip install triton==3.0.0

# Critical flags to bypass the hardware check during Docker build
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"
# Arch 12.0 is specifically for NVIDIA Blackwell (RTX 6000 / RTX 50 series)
ENV TORCH_CUDA_ARCH_LIST="12.0"

# Install SageAttention from source (2.2.0 has native Blackwell support, no patching needed)
RUN pip install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation

# Clone ComfyUI and install dependencies
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app && \
    pip install -r /app/requirements.txt

WORKDIR /app

# Reinstall PyTorch to ensure version stays at 2.7.1
#RUN pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI Manager
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git /app/custom_nodes/ComfyUI-Manager && \
    pip install -r /app/custom_nodes/ComfyUI-Manager/requirements.txt

# Install xformers and additional useful packages
RUN pip install \
    xformers==0.0.31.post1 \
    huggingface-hub[cli] \
    opencv-python \
    scipy \
    einops \
    transformers \
    accelerate \
    safetensors \
    omegaconf

# Install Basic Auth custom node (set COMFYUI_USERNAME and COMFYUI_PASSWORD env vars to enable)
RUN git clone https://github.com/fofr/comfyui-basic-auth.git /app/custom_nodes/comfyui-basic-auth

# Copy and set entrypoint script (ensures base custom nodes exist when custom_nodes is mounted)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ComfyUI port
EXPOSE 8188

# Set working directory to ComfyUI
WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--use-sage-attention"]
