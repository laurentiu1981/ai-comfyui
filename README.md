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

### create_refmod.sh

Build a MiniMax H3 RefMod from a folder of images (identity) and/or videos (motion)
inside the running `comfyui` container. The result lands in `models/refmods/` and
shows up in the **Load H3 RefMods** dropdown
([creation guide](https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)).

**Usage:**

```bash
scripts/create_refmod.sh <folder> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `<folder>` | Yes | Folder with images (png/jpg/jpeg/webp/bmp) and/or videos (mp4/mov/webm/mkv/avi/gif); must be under `./input` |
| `--name` | No | RefMod name (default: `minimaxh3_<folder>_v1_refmod`) |
| `--mode` | No | `encode` (full-res, carries identity/motion) or `training` (pooled, cheap, concept only); default `encode` |
| `--concept-type` | No | `identity`, `pose_motion`, `clothing`, `style`, ... (default: `identity` with images, `pose_motion` for video-only) |
| `--resolution` | No | Short-edge size, multiple of 32 (default: 1024; use 512 or lower for clips) |
| `--max-tokens` | No | Token budget, 0 = unlimited; extra frames are dropped (default: 13500 for images, 0 when videos are present) |
| `--max-frames` | No | Frames sampled per video before encoding; longer clips are sampled uniformly down to this (default: 61) |
| `--all-frames` | No | Keep every frame of every clip (counts them with `ffprobe`) |
| `--latent-frames` | No | Frames kept per video (default: same as `--max-frames`) |
| `--multiplier` | No | Repeat the stacked refs N times along time (default: 1) |
| `--description` | No | Text stored in the file; loaders emit it as `prompt_hint` |
| `--device` | No | `cpu`, `cuda`, `auto` (default: cpu) |
| `--dry-run` | No | List sources and the docker command without encoding |

**Examples:**

```bash
# 8-20 reference images in input/andre -> models/refmods/minimaxh3_andre_v1_refmod.safetensors
scripts/create_refmod.sh input/andre

# One short action clip as a motion reference (lower res keeps the token count sane)
scripts/create_refmod.sh input/wave --resolution 512 --description "waves with the right hand, then turns to camera"

# Same, but keep every frame instead of sampling down to 61
scripts/create_refmod.sh input/wave --resolution 512 --all-frames

# Cheaper identity reference that fits the guide's 8192 budget
scripts/create_refmod.sh input/andre --resolution 768 --max-tokens 8192

# Check which sources will be used first
scripts/create_refmod.sh input/andre --dry-run
```

**Notes:**
- Default device is `cpu` (a `docker exec` python usually can't init CUDA on WSL2 while
  the server holds the GPU); ~11 images take a few minutes.
- All sources are cover-cropped to the first source's aspect ratio; keep one aspect per
  folder. Lower `--resolution` for clips, token cost grows with frames x pixels.
- Re-running with the same name overwrites the file.

### outpaint.sh

Outpaint every photo in a folder to a target aspect ratio with Qwen Image 2.1,
through the running `comfyui` container. One axis keeps its size, the other is
extended, and the original pixels are pasted back untouched. Images already at the
ratio (within a tolerance) are skipped; a failing image is logged and the run goes on.

**Usage:**

```bash
scripts/outpaint.sh <folder> --aspect-ratio <ratio> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `<folder>` | Yes | Folder with images (png/jpg/jpeg/webp/bmp); must be under `./input` |
| `--aspect-ratio`, `-a` | Yes | Target width:height as `1:1`, `16:9`, `4/5`, `3x2` or a decimal like `1.5` |
| `--output`, `-o` | No | Relative path = subfolder of `./output`; absolute or `~` path = copied there (default: `outpainting/{date}`) |
| `--tolerance` | No | Skip images within this percentage of the target ratio (default: 1) |
| `--anchor` | No | Position of the original on the extended axis: `center`, `start`, `end` (default: `center`) |
| `--max-megapixels` | No | Canvas size the model works at; larger canvases are generated smaller and upscaled (default: 4) |
| `--method`, `-m` | No | `lora` (default, outpaint LoRA) or `controlnet` (Fun ControlNet Union inpaint) |
| `--prompt` | No | Fixed scene description instead of the automatic per-image caption |
| `--no-caption` | No | Skip the caption (faster) |
| `--show-prompt` | No | Print the prompt used for each image |
| `--full` | No | 25 steps without the 8-step LoRA (default: 8 steps with it) |
| `--steps`, `--cfg`, `--sampler`, `--scheduler`, `--seed` | No | Sampler settings (defaults: 8, 1, `euler`, `simple`, random seed) |
| `--outpaint-lora`, `--outpaint-lora-strength` | No | `lora`: LoRA file and strength (default: `qwen-image-2.1/qwen-image-2.1-outpaint-v2`, 1.0) |
| `--controlnet`, `--controlnet-strength`, `--no-reference` | No | `controlnet`: patch file, strength, drop the reference image |
| `--lora`, `--model`, `--text-encoder`, `--vae` | No | Extra model-only LoRA and model files |
| `--limit`, `--timeout`, `--url` | No | First N images only; max seconds per image (900); server URL |
| `--dry-run` | No | Print the plan without generating |

Output path placeholders: `{date}` (YYYY-MM-DD), `{datetime}` (YYYY-MM-DD_HH-mm-ss),
`{time}`, `{year}`, `{month}`, `{day}`, `{folder}` (input folder name), `{aspect}`
(`1:1` -> `1x1`).

**Examples:**

```bash
# Square versions of every photo -> output/outpainting/2026-10-30/
scripts/outpaint.sh input/cecile2 --aspect-ratio=1:1

# 16:9 into a per-folder, timestamped subfolder
scripts/outpaint.sh input/cecile2 --aspect-ratio 16:9 --output "outpainting/{folder}/{aspect}/{datetime}"

# Copy the results to a host folder, full 25-step sampling
scripts/outpaint.sh input/cecile2 --aspect-ratio 1.5 --output ~/my-folder --full

# Keep the original at the top, 3% tolerance, ControlNet method
scripts/outpaint.sh input/cecile2 --aspect-ratio 4:5 --anchor start --tolerance 3 --method controlnet

# Check sizes and padding first
scripts/outpaint.sh input/cecile2 --aspect-ratio 1:1 --dry-run
```

**Notes:**
- Runs on the host (`python3` + Pillow) against the running server; credentials from `.env`.
- Needs the Qwen Image 2.1 int8 model, text encoder and VAE, the
  [outpaint LoRA](https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA) and the
  8-step LoRA in `models/loras/qwen-image-2.1/`. `--method controlnet` also needs the
  Fun ControlNet Union patch and the node in `custom_nodes/comfyui-outpaint-helpers`.
- About 10-30 s per image; `--full` and `controlnet` roughly double that.

### outfit.sh

Dress the person in a photo, or in every photo of a folder, in the outfit from another
picture with Qwen Image 2.1 and the
[Outfit Swap Consistency LoRA](https://huggingface.co/ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA),
through the running `comfyui` container. Nothing to mask: the person is reference image 1,
the outfit image 2, and the LoRA keeps face, hair, pose, framing and background in place so
only the clothes change. It is the API-format port of the author's workflow
(`workflows/qwen-image-2.1-outfit-swap.json`), built from core nodes only.

**Usage:**

```bash
scripts/outfit.sh <folder | image> --outfit <image> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `<folder \| image>` | Yes | Image file, or folder with images (png/jpg/jpeg/webp/bmp); must be under `./input` |
| `--outfit` | Yes | Picture of the outfit (a person wearing it, flat lay or product shot); must be under `./input` |
| `--output`, `-o` | No | Relative path = subfolder of `./output`; absolute or `~` path = copied there (default: `outfit/{date}/{outfit}`) |
| `--who` | No | Who to dress in a group photo, e.g. `"the woman on the left"` (default: `the person`) |
| `--prompt` | No | Fixed outfit phrase, e.g. `"the red leather jacket shown in image 2"`, instead of the automatic description |
| `--no-caption` | No | Skip the description and ask for `the outfit shown in image 2` (faster) |
| `--coverage` | No | What the describer covers: `auto` (asks the text encoder whether legs/feet are visible), `full`, `no-shoes`, `top` |
| `--extra` | No | Sentence appended to the request (default: `Replace everything they wear.`; try `"Keep their own shoes and legwear."`) |
| `--template` | No | The whole request with `{who}`, `{outfit}`, `{extra}` placeholders |
| `--show-prompt` | No | Print the request sent to Qwen for each image |
| `--max-megapixels` | No | The model works at the source size (multiples of 32); larger sources are downscaled to this and the result upscaled back (default: 4) |
| `--megapixels`, `--outfit-megapixels` | No | Force a fixed working size for the person picture (the author uses 2); outfit picture size (default: 1) |
| `--model-size` | No | Keep the result at the model's working size instead of scaling it back when they differ |
| `--full` | No | 25 steps without a speed LoRA, the author's tested recipe (default: 8 steps with one) |
| `--speed-lora` | No | `8step` (default, `p_qwen_image_2.1_8step_v0.1`), `viggle` (Viggle Turbo 6-step, the one the author tested) or a file under `models/loras` |
| `--steps`, `--cfg`, `--sampler`, `--scheduler`, `--seed` | No | Sampler settings (defaults: 8, 1, `euler`, `simple`, random seed) |
| `--outfit-lora`, `--outfit-lora-strength`, `--no-outfit-lora` | No | LoRA file and strength (default: `qwen-image-2.1/qwen-image-2.1-outfit-swap`, 1.0), or plain Qwen |
| `--no-cache`, `--lora`, `--model`, `--text-encoder`, `--vae` | No | Drop the QwenImage21Cache node; extra model-only LoRA and model files |
| `--limit`, `--timeout`, `--url` | No | First N images only; max seconds per image (900); server URL |
| `--dry-run` | No | Print the plan without generating |

Output path placeholders: `{date}` (YYYY-MM-DD), `{datetime}` (YYYY-MM-DD_HH-mm-ss),
`{time}`, `{year}`, `{month}`, `{day}`, `{folder}` (input folder name), `{outfit}`
(outfit file name without extension).

**Examples:**

```bash
# Every photo of a folder in the dress -> output/outfit/2026-11-08/rochie_gym/
scripts/outfit.sh input/cecile2 --outfit input/rochie_gym.webp

# One photo, print the request Qwen got
scripts/outfit.sh input/cecile2/photo.png --outfit input/rochie_gym.webp --show-prompt

# Group photo, full 25-step sampling at the author's fixed 2 MP working size
scripts/outfit.sh input/party --outfit input/jacket.png --who "the woman on the left" --full --megapixels 2

# 8 steps with the Viggle Turbo LoRA the author tested
scripts/outfit.sh input/cecile2 --outfit input/rochie_gym.webp --speed-lora viggle

# Your own outfit phrase, keep their shoes
scripts/outfit.sh input/cecile2 --outfit input/jacket.png --prompt "the black leather jacket shown in image 2" --extra "Keep their own shoes and legwear."

# Per-folder, per-outfit, timestamped subfolder; first 3 images only
scripts/outfit.sh input/cecile2 --outfit input/jacket.png --output "outfit/{folder}/{outfit}/{datetime}" --limit 3
```

**Notes:**
- Runs on the host (`python3` + Pillow) against the running server; credentials from `.env`.
- Needs the Qwen Image 2.1 int8 model, text encoder and VAE, the outfit swap LoRA and the
  8-step LoRA in `models/loras/qwen-image-2.1/`; `--speed-lora viggle` needs
  [Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) there too
  (Qwen Research License, research use).
- The outfit description comes from the Qwen3-VL text encoder: as in the original workflow it
  first checks whether the person picture shows legs and feet, and leaves shoes or the lower
  half out of the description so the picture is not zoomed out to fit the whole outfit.
  Crop the outfit picture to the clothes you want when it shows more.
- A source with sides that are multiples of 32 and under 4 MP is never resampled; otherwise
  the only resample is to the nearest multiple of 32 and back (or down to the cap and up).
- About 15 s per image on the RTX PRO 6000, `--full` about the same (the description
  dominates); `--no-caption` or `--prompt` skips the three text-encoder passes.
- Limits per the author: shoes are stubborn, the pose is kept (an outfit that needs
  another pose will not get one), and the outfit is not drawn more exactly than plain Qwen.
