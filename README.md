# ComfyUI Docker (Blackwell + SageAttention 2.2)

Dockerized ComfyUI optimized for NVIDIA Blackwell GPUs (RTX PRO 6000, RTX 50 series).

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA Blackwell GPU with driver supporting CUDA 12.8+

## Quick Start

```bash
./setup-dirs.sh
# Place models in ./models/
docker compose build
docker compose up -d
```

Access at http://localhost:8188

## Directory Structure

```
models/
├── checkpoints/    # SD checkpoints
├── vae/
├── loras/
├── controlnet/
├── clip/
├── unet/           # Flux, SD3, etc.
├── embeddings/
├── upscale_models/
└── clip_vision/
input/              # Input images
output/             # Generated images
```

## Stack

- CUDA 12.8 / Python 3.13 / PyTorch 2.7.1
- SageAttention 2.2
- ComfyUI + ComfyUI-Manager
- xformers
