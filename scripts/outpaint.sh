#!/usr/bin/env bash
# Outpaint every photo in a folder to a target aspect ratio with Qwen Image 2.1,
# through the running comfyui container's API (no resize of the original: one
# axis is kept, the other is extended and the original pixels are pasted back).
#
# The folder must live under this repo's ./input directory, which is what the
# container sees as /app/input. Results land in ./output/<path> (default
# ./output/outpainting/<YYYY-MM-DD>); absolute output paths are copied there.
#
# Usage:
#   scripts/outpaint.sh <folder> --aspect-ratio <W:H | W/H | WxH | decimal> [options]
#
# Examples:
#   scripts/outpaint.sh input/cecile2 --aspect-ratio=1:1
#   scripts/outpaint.sh input/cecile2 --aspect-ratio 16:9 --output "outpainting/cecile/{date}"
#   scripts/outpaint.sh input/cecile2 --aspect-ratio 1.5 --output ~/my-folder --full
#   scripts/outpaint.sh input/cecile2 --aspect-ratio 4:5 --dry-run
#
# All options: scripts/outpaint.sh --help
#
# The heavy lifting is in scripts/outpaint.py (host python3 + Pillow; the
# ComfyUI server does the generation, so the GPU is used by the server as usual).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v python3 >/dev/null || { echo "error: python3 is required on the host" >&2; exit 1; }
python3 -c 'import PIL' 2>/dev/null || { echo "error: Pillow is required on the host (pip install pillow)" >&2; exit 1; }

exec python3 "$here/outpaint.py" "$@"
