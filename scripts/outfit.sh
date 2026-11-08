#!/usr/bin/env bash
# Dress the person in a photo (or in every photo of a folder) in the outfit from
# another picture, with Qwen Image 2.1 and the ausboss Outfit Swap Consistency
# LoRA, through the running comfyui container's API. Nothing to mask: the model
# gets the person as image 1, the outfit as image 2 and a request; the LoRA keeps
# the face, pose, framing and background in place so only the clothes change.
#
# Both the source and the outfit picture must live under this repo's ./input
# directory, which is what the container sees as /app/input. Results land in
# ./output/<path> (default ./output/outfit/<YYYY-MM-DD>/<outfit-name>); absolute
# output paths are copied there.
#
# Usage:
#   scripts/outfit.sh <folder | image> --outfit <image> [options]
#
# Examples:
#   scripts/outfit.sh input/cecile2 --outfit input/rochie_gym.webp
#   scripts/outfit.sh input/cecile2/photo.png --outfit input/rochie_gym.webp --show-prompt
#   scripts/outfit.sh input/cecile2 --outfit input/jacket.png --who "the woman on the left" --full --megapixels 2
#   scripts/outfit.sh input/cecile2 --outfit input/jacket.png --prompt "the black leather jacket shown in image 2"
#   scripts/outfit.sh input/cecile2 --outfit input/jacket.png --output "outfit/{folder}/{outfit}/{datetime}" --limit 3
#
# All options: scripts/outfit.sh --help
#
# The heavy lifting is in scripts/outfit.py (host python3 + Pillow; the ComfyUI
# server does the generation, so the GPU is used by the server as usual).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v python3 >/dev/null || { echo "error: python3 is required on the host" >&2; exit 1; }
python3 -c 'import PIL' 2>/dev/null || { echo "error: Pillow is required on the host (pip install pillow)" >&2; exit 1; }

exec python3 "$here/outfit.py" "$@"
