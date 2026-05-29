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

# Run the main command
exec "$@"
