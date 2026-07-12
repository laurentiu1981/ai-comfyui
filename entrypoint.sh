#!/bin/bash
set -e

# Ensure base custom nodes are present (they're baked into the image
# but get hidden when custom_nodes is mounted as a host volume)

if [ ! -d "/app/custom_nodes/ComfyUI-Manager" ]; then
    echo "Installing ComfyUI-Manager..."
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git /app/custom_nodes/ComfyUI-Manager
    pip install -r /app/custom_nodes/ComfyUI-Manager/requirements.txt
fi

if [ ! -d "/app/custom_nodes/comfyui-basic-auth" ]; then
    echo "Installing comfyui-basic-auth..."
    git clone https://github.com/fofr/comfyui-basic-auth.git /app/custom_nodes/comfyui-basic-auth
fi

# Re-apply vendored ComfyUI-Manager patches (see PATCHES.md). The Manager lives in
# the host-mounted custom_nodes/, so updating it silently reverts local fixes; this
# re-applies them on every boot. Idempotent: skips if already applied, warns if the
# patch no longer matches the source (Manager updated past it).
for patch in /patches/comfyui-manager-*.patch; do
    [ -f "$patch" ] || continue
    if git -C /app/custom_nodes/ComfyUI-Manager apply --reverse --check "$patch" 2>/dev/null; then
        : # already applied
    elif git -C /app/custom_nodes/ComfyUI-Manager apply "$patch" 2>/dev/null; then
        echo "Applied ComfyUI-Manager patch: $(basename "$patch")"
    else
        echo "WARNING: $(basename "$patch") no longer applies cleanly; check if still needed (PATCHES.md)"
    fi
done

# Install requirements only for custom nodes that haven't been installed yet.
# Stamp files live in site-packages so they're removed if the volume is reset.
STAMP_DIR="/usr/local/lib/python3.12/dist-packages/.node_stamps"
mkdir -p "$STAMP_DIR"

nodes_changed=0
for req in /app/custom_nodes/*/requirements.txt; do
    [ -f "$req" ] || continue
    node_name=$(basename "$(dirname "$req")")
    current_hash=$(md5sum "$req" | cut -d' ' -f1)
    stamp_file="$STAMP_DIR/$node_name"

    if [ -f "$stamp_file" ] && [ "$(cat "$stamp_file")" = "$current_hash" ]; then
        continue
    fi

    echo "Installing requirements for $node_name..."
    pip install -r "$req" && echo "$current_hash" > "$stamp_file" && nodes_changed=1
done

# Re-assert ComfyUI's own pins LAST so they win over anything a custom node
# downgraded (e.g. comfyui_layerstyle -> inference-gpu pins av<13, but ComfyUI
# needs av>=16). Runs when a node was (re)installed this boot or when core
# requirements changed (image rebuild / volume reset). Skipped on a clean restart.
CORE_STAMP="$STAMP_DIR/.comfyui_core"
core_hash=$(md5sum /app/requirements.txt | cut -d' ' -f1)
if [ "$nodes_changed" = "1" ] || [ "$(cat "$CORE_STAMP" 2>/dev/null)" != "$core_hash" ]; then
    echo "Re-asserting ComfyUI core requirements..."
    pip install -r /app/requirements.txt && echo "$core_hash" > "$CORE_STAMP"
fi

# ComfyUI-Trellis2 needs extras its requirements.txt can't express (see PATCHES.md
# 2026-07-12): bundled CUDA wheels, CUDA-12 runtime libs (the wheels link
# cudart/nvrtc 12 while the image is CUDA 13), nvdiffrast rebuilt from source to
# match the image's torch ABI, and transformers>=4.56 (DINOv3ViTModel).
# Runs only while the node is present; re-runs after a volume reset (stamp lives
# in site-packages) or when the node's bundled wheels change.
TRELLIS_WHEELS="/app/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch291" # cp312 = image python
if [ -d "$TRELLIS_WHEELS" ]; then
    stamp_file="$STAMP_DIR/ComfyUI-Trellis2.extras"
    current_hash=$(ls "$TRELLIS_WHEELS" | md5sum | cut -d' ' -f1)
    if [ ! -f "$stamp_file" ] || [ "$(cat "$stamp_file")" != "$current_hash" ]; then
        echo "Installing ComfyUI-Trellis2 extras (CUDA wheels, cu12 libs, nvdiffrast, transformers)..."
        pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 plyfile zstandard \
        && pip install --no-deps "$TRELLIS_WHEELS"/cumesh-*.whl \
               "$TRELLIS_WHEELS"/flex_gemm-*.whl "$TRELLIS_WHEELS"/nvdiffrec_render-*.whl \
        && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" pip install --no-build-isolation \
               --force-reinstall --no-deps \
               'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0' \
               'o_voxel @ git+https://github.com/visualbruno/TRELLIS.2.git#subdirectory=o-voxel' \
        && { python -c 'from transformers import DINOv3ViTModel' 2>/dev/null \
             || pip install 'transformers==4.56.2'; } \
        && echo "$current_hash" > "$stamp_file" \
        || echo "WARNING: ComfyUI-Trellis2 extras install failed; the node may not load (see PATCHES.md)"
    fi
fi

# Run the main command
exec "$@"
