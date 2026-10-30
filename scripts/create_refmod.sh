#!/usr/bin/env bash
# Build a MiniMax H3 RefMod from a folder of reference images and/or videos,
# inside the running comfyui container (ComfyUI-MiniMaxH3Mod's extract_mod.py).
#
# The folder must live under this repo's ./input directory, which is what the
# container sees as /app/input. Output goes to models/refmods/ (host:
# ~/ai/models/refmods), where the Load H3 RefMods node picks it up.
# Images and videos in the same folder are stacked into one RefMod; every
# source is cover-cropped to the first source's aspect ratio (images first).
#
# Usage:
#   scripts/create_refmod.sh <folder> [options]
#
# Options:
#   --name <name>          RefMod name (default: minimaxh3_<folder>_v1_refmod)
#   --mode <mode>          encode | training (default: encode; training = pooled
#                          concept/motion grid, cheap but loses identity)
#   --concept-type <type>  identity | pose_motion | clothing | background | style |
#                          generic ... (default: identity if the folder has images,
#                          pose_motion if it has only videos)
#   --resolution <px>      short-edge resolution, multiple of 32 (default: 1024)
#   --max-tokens <n>       token budget, 0 = unlimited; frames beyond it are
#                          dropped (default: 13500 for images, 0 when videos are
#                          present so motion timing is not thinned out)
#   --max-frames <n>       frames sampled per video before encoding (default: 61;
#                          longer clips are sampled uniformly down to this)
#   --all-frames           keep every frame of every clip (reads the frame count
#                          with ffprobe and sets --max-frames to the longest clip)
#   --latent-frames <n>    frames kept per video; encode mode samples to this,
#                          training mode pools to it (default: same as --max-frames)
#   --multiplier <n>       repeat the stacked refs N times along time (default: 1)
#   --description <text>   stored in the file; loaders emit it as prompt_hint
#   --device <dev>         cpu | cuda | auto (default: cpu — see note below)
#   --dry-run              list the sources and the command, encode nothing
#
# Examples:
#   scripts/create_refmod.sh input/andre                       # 8-20 stills
#   scripts/create_refmod.sh input/wave --resolution 512       # one action clip
#   scripts/create_refmod.sh input/wave --resolution 512 --all-frames
#   scripts/create_refmod.sh input/andre_dance --max-tokens 20000
#
# Token cost: encode mode costs (H/32)*(W/32) tokens per latent frame (~1200 at
# 1024x1216) and the VAE compresses time 4:1, so a 61-frame clip at 1024px is
# ~16 latent frames or ~19k tokens. Lower --resolution for clips.
#
# Note on --device: on this WSL2 host a python started via `docker exec` often
# cannot initialise CUDA while the ComfyUI server holds the GPU, so the default
# is cpu. ~11 images at 1024px take a few minutes; clips take longer.

set -euo pipefail

CONTAINER="${CONTAINER:-comfyui}"
VAE="/app/models/vae/minimax_h3_video_vae_fp16.safetensors"
EXTRACT="custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py"

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ "$1" == "-h" || "$1" == "--help" ]] && usage 0

folder="$1"; shift
name="" mode=encode concept="" resolution=1024 max_tokens="" max_frames=61 latent_frames=""
multiplier=1 description="" device=cpu dry_run=0 all_frames=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)          name="$2"; shift 2 ;;
        --mode)          mode="$2"; shift 2 ;;
        --concept-type)  concept="$2"; shift 2 ;;
        --resolution)    resolution="$2"; shift 2 ;;
        --max-tokens)    max_tokens="$2"; shift 2 ;;
        --max-frames)    max_frames="$2"; shift 2 ;;
        --all-frames)    all_frames=1; shift ;;
        --latent-frames) latent_frames="$2"; shift 2 ;;
        --multiplier)    multiplier="$2"; shift 2 ;;
        --description)   description="$2"; shift 2 ;;
        --device)        device="$2"; shift 2 ;;
        --dry-run)       dry_run=1; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="$repo_dir/input"

[[ -d "$folder" ]] || { echo "error: '$folder' is not a directory" >&2; exit 1; }
folder="$(cd "$folder" && pwd)"
case "$folder" in
    "$input_dir"/*) rel="${folder#"$input_dir"/}" ;;
    *) echo "error: folder must be inside $input_dir (the container only sees ./input)" >&2; exit 1 ;;
esac

if [[ -z "$name" ]]; then
    base="$(basename "$folder" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/_/g; s/^_//; s/_$//')"
    name="minimaxh3_${base}_v1_refmod"
fi

mapfile -t images < <(find "$folder" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \) \
    | sort)
mapfile -t videos < <(find "$folder" -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.gif' \) \
    | sort)
[[ ${#images[@]} -gt 0 || ${#videos[@]} -gt 0 ]] || {
    echo "error: no images (png/jpg/jpeg/webp/bmp) or videos (mp4/mov/webm/mkv/avi/gif) in $folder" >&2; exit 1; }

# Content-dependent defaults (all overridable above)
if [[ -z "$concept" ]]; then
    [[ ${#images[@]} -gt 0 ]] && concept=identity || concept=pose_motion
fi
if [[ -z "$max_tokens" ]]; then
    [[ ${#videos[@]} -gt 0 ]] && max_tokens=0 || max_tokens=13500
fi
if [[ $all_frames -eq 1 && ${#videos[@]} -gt 0 ]]; then
    command -v ffprobe >/dev/null || { echo "error: --all-frames needs ffprobe on the host" >&2; exit 1; }
    max_frames=0
    for f in "${videos[@]}"; do
        n="$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$f")"
        [[ "$n" =~ ^[0-9]+$ ]] || { echo "error: could not count frames in $f" >&2; exit 1; }
        echo "Frames      : $n in ${f##*/}"
        (( n > max_frames )) && max_frames=$n
    done
    latent_frames="$max_frames"
fi
[[ -z "$latent_frames" ]] && latent_frames="$max_frames"

args=(python "$EXTRACT"
    --vae "$VAE"
    --name "$name"
    --mode "$mode"
    --concept-type "$concept"
    --resolution "$resolution"
    --max-tokens "$max_tokens"
    --max-frames "$max_frames"
    --latent-frames "$latent_frames"
    --multiplier "$multiplier"
    --device "$device")
[[ -n "$description" ]] && args+=(--description "$description")
for f in "${images[@]}"; do args+=(--image "/app/input/$rel/$(basename "$f")"); done
for f in "${videos[@]}"; do args+=(--video "/app/input/$rel/$(basename "$f")"); done

echo "RefMod name : $name"
echo "Sources     : ${#images[@]} image(s), ${#videos[@]} video(s) in input/$rel"
echo "Settings    : mode=$mode concept=$concept resolution=$resolution max_tokens=$max_tokens device=$device"
[[ ${#videos[@]} -gt 0 ]] && echo "Video       : max_frames=$max_frames latent_frames=$latent_frames multiplier=$multiplier"

if [[ $dry_run -eq 1 ]]; then
    printf '  %s\n' "${images[@]##*/}" "${videos[@]##*/}"
    echo; echo "docker exec $CONTAINER $(printf '%q ' "${args[@]}")"
    exit 0
fi

# Keep the container's python warnings (pynvml/urllib3) out of the way.
docker exec "$CONTAINER" "${args[@]}" 2>&1 | grep -v -E 'FutureWarning|RequestsDependencyWarning|warnings\.warn|import pynvml'
