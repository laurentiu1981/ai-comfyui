# ComfyUI Docker (Blackwell + SageAttention 2.2)

Dockerized ComfyUI optimized for NVIDIA Blackwell GPUs (RTX PRO 6000, RTX 50 series).

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA Blackwell GPU with driver supporting CUDA 13.0+

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

- CUDA 13.0 / Python 3.12 / PyTorch 2.10.0
- SageAttention 2.2
- ComfyUI + ComfyUI-Manager
- xformers

## Upgrading the Base Image

When upgrading to a new PyTorch base image (e.g. newer CUDA or PyTorch version), follow these steps:

### 1. Update `Dockerfile`

Change the `FROM` line to the new image tag:
```dockerfile
FROM pytorch/pytorch:<version>-cuda<cuda_version>-cudnn9-devel
```

Check [Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags) for available tags.

### 2. Check the Python version and site-packages path

The new image may use a different Python version or package path:
```bash
docker run --rm pytorch/pytorch:<new-tag> python -c "import sys; import site; print(f'Python {sys.version}'); print(site.getsitepackages()[0])"
```

### 3. Update paths if Python version changed

If the site-packages path changed, update it in two places:

- **`docker-compose.yml`** — the `site_packages` volume mount:
  ```yaml
  - site_packages:/usr/local/lib/python<VERSION>/dist-packages
  ```

- **`entrypoint.sh`** — the `STAMP_DIR` path:
  ```bash
  STAMP_DIR="/usr/local/lib/python<VERSION>/dist-packages/.node_stamps"
  ```

### 4. Check for Ubuntu/package changes

Newer base images may use a different Ubuntu version. Common issues:
- `libgl1-mesa-glx` → replaced by `libgl1` in Ubuntu 24.04+
- `PIP_BREAK_SYSTEM_PACKAGES=1` — needed if the image uses PEP 668 managed Python

### 5. Reset the site_packages volume and rebuild

```bash
docker compose down
docker volume rm <project>_site_packages   # e.g. ai-comfyui_site_packages
docker compose build
docker compose up -d
```

The named volume must be removed so Docker re-initializes it from the new image. Custom node code in `./custom_nodes/` is unaffected — their pip deps will be reinstalled automatically on first start.

## Scripts

### rename_files.py

Bulk rename files with sequential counters starting from a given index.

**Usage:**

```bash
python3 scripts/rename_files.py --folder=<path> --prefix=<prefix> --index=<start_index>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--folder`, `-f` | Yes | Folder containing files (relative or absolute path) |
| `--prefix`, `-p` | Yes | File prefix to match (e.g., `ComfyUI_`) |
| `--index`, `-i` | Yes | Starting index for renaming |
| `--suffix`, `-s` | No | Suffix after counter (auto-detected if not provided) |
| `--dry-run`, `-d` | No | Preview changes without renaming |

**Examples:**

```bash
# Rename ComfyUI_00001_.png, ComfyUI_00002_.png, ... starting from index 2300
python3 scripts/rename_files.py --folder=./output --prefix=ComfyUI_ --index=2300

# Preview what would be renamed (dry run)
python3 scripts/rename_files.py --folder=./output --prefix=ComfyUI_ --index=2300 --dry-run

# Specify a custom suffix
python3 scripts/rename_files.py --folder=./output --prefix=image --index=100 --suffix=_
```

**Notes:**
- The script auto-detects the suffix between the counter and file extension
- Counter width is preserved (e.g., 5-digit counters stay 5 digits)
- Shows statistics and handles permission errors with suggestions

### prepend_text.py

Prepend text to all `.txt` files in a folder. Useful for adding trigger words to training caption files.

**Usage:**

```bash
python3 scripts/prepend_text.py --folder=<path> --text="<text to prepend>"
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--folder`, `-f` | Yes | Folder containing .txt files (relative or absolute path) |
| `--text`, `-t` | Yes | Text to prepend to each file |
| `--dry-run`, `-d` | No | Preview changes without modifying files |
| `--skip-empty` | No | Skip empty .txt files |
| `--skip-existing` | No | Skip files that already start with the prepend text |

**Examples:**

```bash
# Add trigger word to all caption files
python3 scripts/prepend_text.py --folder=./input/dataset --text="[trigger], "

# Preview changes first
python3 scripts/prepend_text.py --folder=./captions --text="photo of sks person, " --dry-run

# Skip files that already have the trigger
python3 scripts/prepend_text.py --folder=./data --text="[trigger], " --skip-existing
```

**Notes:**
- Files are read and written as UTF-8
- Use `--skip-existing` to safely re-run without duplicating the prepend text
- Shows statistics and handles permission/encoding errors
