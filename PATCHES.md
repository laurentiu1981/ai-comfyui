# Patches & Fixes Log

Running record of out-of-band fixes applied to this ComfyUI deployment — things
that aren't captured by the `Dockerfile` / `docker-compose.yml` alone and may
need to be re-applied after certain operations (image rebuilds, volume resets,
custom-node updates).

Context for all entries below:

- Image: `cracoidic/crk-comfyui-blackwell:pytorch-2.10.0-cuda13.0`
- Package volume: `ai-comfyui_site_packages` mounted at `/usr/local/lib/python3.12/dist-packages`
- ComfyUI source lives in the image at `/app` (NOT a volume)
- `custom_nodes/` is a host-mounted directory (changes there persist on the host)

> **Key gotcha that caused several of today's issues:** a Docker **named volume is
> only seeded from the image the first time it's created (while empty)**. After
> that, the volume's contents *shadow* whatever the image ships. So rebuilding the
> image updates `/app` (baked in) but does **not** update Python packages in the
> `site_packages` volume. Image + volume can silently drift out of sync.

---

## 2026-05-29 — Session after rebuilding the image to update ComfyUI + Manager

Four issues surfaced after an image rebuild. Summary table, details below.

| # | Symptom | Root cause | Fix | Survives volume reset? |
|---|---------|-----------|-----|------------------------|
| 1 | `ModuleNotFoundError: No module named 'comfy_aimdo.host_buffer'` at startup | Stale `site_packages` volume held old `comfy-aimdo 0.2.8`; new `/app` code needs `0.4.5` | Reinstall ComfyUI requirements into the volume | Yes — image has 0.4.5, self-heals |
| 2 | `10s-comfy-nodes` "looked" un-installed in Manager | Not actually broken — normal Manager UI + lowercase folder name | None needed | n/a |
| 3 | `Failed to execute install script: comfyui_nvidia_rtx_nodes` → `No module named 'nvvfx'` | `nvidia-vfx` PyPI entry is a stub; real wheel is on NVIDIA's index | Install from NVIDIA index + add `PIP_EXTRA_INDEX_URL` env var | Yes — env var makes entrypoint reinstall work |
| 4 | `ComfyUI-LTXVideo` IMPORT FAILED: `cannot import name 'pad' from 'kornia.geometry.transform.pyramid'` | kornia 0.8.3 removed the `pad` re-export the node relied on | Applied upstream PR #498 (use `F.pad`) to the node source | Yes (lives in `custom_nodes/`) — unless node is updated |

---

### Issue 1 — `comfy_aimdo.host_buffer` ModuleNotFoundError

**Symptom (startup crash):**
```
File "/app/comfy/model_management.py", line 35, in <module>
    import comfy_aimdo.host_buffer
ModuleNotFoundError: No module named 'comfy_aimdo.host_buffer'
```

**Root cause:** ComfyUI master now pins `comfy-aimdo==0.4.5` and `comfy-kitchen==0.2.9`
in `requirements.txt`, and `model_management.py` imports `comfy_aimdo.host_buffer`.
The rebuild updated `/app` (new code) but the `site_packages` volume still held the
**old** `comfy-aimdo 0.2.8` (no `host_buffer` submodule). New code + old package = crash.

**Fix — re-sync the volume's packages to the image's requirements:**
```bash
docker compose exec comfyui pip install -r /app/requirements.txt
docker compose restart comfyui
```
This upgraded `comfy-aimdo 0.2.8 → 0.4.5`, `comfy-kitchen 0.2.7 → 0.2.9`, and the
frontend packages.

**Durability:** The rebuilt image already contains `0.4.5`, so a clean volume reset
(`docker volume rm ai-comfyui_site_packages` + `docker compose up -d`) re-seeds correctly.
Re-run the re-sync command above any time you rebuild the image to a newer ComfyUI
while keeping an existing volume.

---

### Issue 2 — `10s-comfy-nodes` (https://github.com/TenStrip/10S-Comfy-nodes)

**First impression (wrong):** Looked like a false alarm — the pack loaded with no
`(IMPORT FAILED)` and registered 13 nodes, and the Manager's update/disable/uninstall
buttons are normal for any installed node.

**Actual problem (found later):** The Manager installed the **Comfy Registry zip**
(`pyproject` version 1.0.0, 13 nodes, no `.git`), but the workflow
`user/default/workflows/crk_ltx2.3-10eros.json` references 10S nodes by
`aux_id: TenStrip/10S-Comfy-nodes` at **specific GitHub commits** that are newer than
the registry zip. So nodes like `LTXFaceDetector`, `LTXLikenessAnchor`,
`LTXLikenessGuide` showed up as **missing** in the workflow — the registry version
simply didn't contain them.

**How to diagnose this class of problem** (node loads fine but workflow shows it missing):
compare the workflow's node `type` values against what the pack registers:
```bash
docker compose exec -T comfyui python - <<'PY'
import json, importlib.util, sys
wf = json.load(open("/app/user/default/workflows/crk_ltx2.3-10eros.json"))
types = sorted({n.get("type","") for n in wf.get("nodes",[])})
spec = importlib.util.spec_from_file_location("tn","/app/custom_nodes/10s-comfy-nodes/__init__.py", submodule_search_locations=["/app/custom_nodes/10s-comfy-nodes"])
m = importlib.util.module_from_spec(spec); sys.modules["tn"]=m; spec.loader.exec_module(m)
reg = set(m.NODE_CLASS_MAPPINGS)
for t in types:
    if t.startswith(("LTX","Latent","Audio")) or "Anchor" in t:
        print(("MISSING " if t not in reg else "ok      ")+t)
PY
```

**Fix — replace the registry zip with the GitHub HEAD clone** (per the repo's README):
```bash
docker compose exec -T comfyui bash -lc '
  cd /app/custom_nodes && rm -rf 10s-comfy-nodes &&
  git clone https://github.com/TenStrip/10S-Comfy-nodes.git 10s-comfy-nodes'
docker compose restart comfyui
```
Result: pack now registers **18 nodes** (was 13), including the three the workflow needed.
GitHub HEAD's `requirements.txt` needs nothing required — only **optional** `mediapipe`
(better face-bbox detection in the Face Detector / Likeness Guide nodes; falls back to
OpenCV Haar cascades otherwise). To enable: `docker compose exec comfyui pip install mediapipe`.

**Durability:** Lives in host-mounted `custom_nodes/`, survives volume resets/rebuilds.
It's now a git clone (not a registry zip), so future updates work via `git pull` /
Manager. **Caveat:** don't let the Manager "reinstall" it from the registry — that would
revert it to the older zip.

> **Note:** the same workflow also needs newer/other packs for non-10S nodes — see the
> "Outstanding" section below. The 10S swap only fixes the 10S-owned nodes.

---

### Issue 3 — `comfyui_nvidia_rtx_nodes` (https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI)

**Symptom:**
```
Installation Error: Failed to execute install script: comfyui_nvidia_rtx_nodes@0.1.3
...
import nvvfx
ModuleNotFoundError: No module named 'nvvfx'
```

**Root cause:** The node's `requirements.txt` is just `nvidia-vfx`, which provides the
`nvvfx` module. But `nvidia-vfx` on public PyPI is a **stub that refuses to install** —
the real (~597 MB) wheel lives on NVIDIA's own package index. The Manager / entrypoint
run plain `pip install -r requirements.txt` with no extra index, so it failed.

**Fix — install from NVIDIA's index:**
```bash
docker compose exec comfyui pip install --extra-index-url https://pypi.nvidia.com/ nvidia-vfx
docker compose restart comfyui
```
Installed `nvidia-vfx 0.1.0.1`; node then loaded cleanly.

**Durability fix (repo change):** Added to `docker-compose.yml` under `environment:` so
pip *always* knows about NVIDIA's index (entrypoint reinstalls, rebuilds, volume resets
all resolve `nvidia-vfx` correctly):
```yaml
      - PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com/
```
> Env var changes only take effect on **container recreation** (`docker compose up -d`),
> not a plain `docker compose restart`.

---

### Issue 4 — `ComfyUI-LTXVideo` (https://github.com/Lightricks/ComfyUI-LTXVideo)

**Symptom (UI: "Import failed"):**
```
File ".../ComfyUI-LTXVideo/pyramid_blending.py", line 7, in <module>
    from kornia.geometry.transform.pyramid import (... pad ...)
ImportError: cannot import name 'pad' from 'kornia.geometry.transform.pyramid'
```

**Root cause:** `pad` was never a real pyramid function — older kornia's `pyramid.py`
just re-exported `torch.nn.functional.pad` (via `kornia.core.pad`). **kornia 0.8.3**
(pulled in by the rebuild) removed that re-export, so the import fails. Confirmed
upstream as **issue #496**; the proper fix is **PR #498**.

**Fix — applied upstream PR #498 to the node source** (`custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py`):
- Removed `pad,` from the `kornia.geometry.transform.pyramid` import.
- Changed the two call sites to use the already-imported `torch.nn.functional as F`:
  - `pad(image, (0, pad_right, 0, pad_down), "reflect")` → `F.pad(image, (0, pad_right, 0, pad_down), mode="reflect")`
  - `pad(images, padding, border_type)` → `F.pad(images, padding, mode=border_type)`

This is functionally identical (the node already used `F.pad` for the mask on the line
that became line 143).

**Why not downgrade kornia:** issue #496 suggests `kornia==0.6.12`, but that's a global,
~2-year-old pin that risks ComfyUI core and other nodes, and wouldn't survive a volume
reset (kornia is unpinned in the node's `requirements.txt`).

**Durability:** Lives in `custom_nodes/` (host-mounted), so it survives volume resets and
rebuilds. **Caveat:** a Manager "update" of this node may overwrite it (the working tree is
now dirty). Re-apply the PR #498 change if that happens — or, once PR #498 is merged
upstream, update the node and drop this patch.

**To re-apply manually if needed** (run from repo root; file is root-owned so edit via container):
```bash
docker compose exec -T comfyui python - <<'PY'
p = "/app/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
s = open(p).read()
reps = [
    ("    find_next_powerof_two,\n    is_powerof_two,\n    pad,\n)",
     "    find_next_powerof_two,\n    is_powerof_two,\n)"),
    ('        image = pad(image, (0, pad_right, 0, pad_down), "reflect")',
     '        image = F.pad(image, (0, pad_right, 0, pad_down), mode="reflect")'),
    ("        images = pad(images, padding, border_type)",
     "        images = F.pad(images, padding, mode=border_type)"),
]
for old, new in reps:
    assert s.count(old) == 1, f"expected 1 match for: {old!r} got {s.count(old)}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("applied PR #498 changes")
PY
docker compose restart comfyui
```

---

### Issue 5 — `LoadImage`: `'av.video.frame.VideoFrame' object has no attribute 'rotation'`

**Symptom (runtime error on a LoadImage node loading a video/animated file):**
```
AttributeError: 'av.video.frame.VideoFrame' object has no attribute 'rotation'
  ... in /app/comfy_api/latest/_input_impl/video_types.py line 308: if frame.rotation != 0
```

**Root cause:** Same **stale-volume shadowing** as Issue 1, but for PyAV (`av`):
- ComfyUI master requires `av>=16.0.0` (uses `VideoFrame.rotation`, added in av 14+).
- Image baked in `av 17.0.1` (correct).
- The `site_packages` volume held a stale `av 12.3.0`, which has no `rotation` attribute.

Why it was stuck at 12.3.0: `comfyui_layerstyle` → `inference-gpu` → `aiortc`, and
`aiortc` requires `av<13.0.0`. When the entrypoint installed layerstyle's requirements,
pip pulled `inference-gpu` and **downgraded av to 12.3.0**, overriding ComfyUI's pin.
(`inference-gpu` also forces the pre-existing numpy/pillow/scikit-image conflicts.)
This is an **unsatisfiable constraint** — ComfyUI needs `av>=16`, inference-gpu needs
`av<13`. ComfyUI must win or LoadImage is broken; the only loss is layerstyle's
Roboflow-inference / WebRTC features (rarely used).

**Fix:**
```bash
docker compose exec comfyui pip install "av>=16.0.0"   # installs av 17.0.1
docker compose restart comfyui
```
Verified: `av 17.0.1`, `VideoFrame.rotation` present, LoadImage works. Persists across
plain restarts (layerstyle's requirements stamp is unchanged, so the entrypoint doesn't
reinstall/downgrade it).

**Durability (hardened in `entrypoint.sh`):** Without a safeguard, a `site_packages`
**volume reset** would make the entrypoint reinstall all custom-node requirements —
layerstyle re-pulls `inference-gpu` and re-downgrades av to <13, re-breaking LoadImage.
Fixed by re-asserting ComfyUI's own requirements **last** in `entrypoint.sh`, after the
custom-node loop, so core pins always win:
```bash
# (end of entrypoint.sh) — runs when a node was (re)installed this boot OR core reqs changed
CORE_STAMP="$STAMP_DIR/.comfyui_core"
core_hash=$(md5sum /app/requirements.txt | cut -d' ' -f1)
if [ "$nodes_changed" = "1" ] || [ "$(cat "$CORE_STAMP" 2>/dev/null)" != "$core_hash" ]; then
    echo "Re-asserting ComfyUI core requirements..."
    pip install -r /app/requirements.txt && echo "$core_hash" > "$CORE_STAMP"
fi
```
(`nodes_changed` is set to 1 inside the node loop whenever a `pip install` runs — this
also catches a future **node update** that downgrades a core package, not just core-reqs
changes.) This also makes Issue 1 (`comfy-aimdo`) self-heal on volume resets.
> **Takes effect after `docker compose build`** — `entrypoint.sh` is baked into the
> image (`COPY entrypoint.sh /entrypoint.sh`). The running container keeps the old
> entrypoint until rebuilt; the live av fix already persists across plain restarts.

---

### Issue 6 — RTX VSR: `NvVFX_Load failed: The effect has not been properly initialized (code -12)`

**Symptom:** Running the `comfyui_nvidia_rtx_nodes` Video Super Resolution node fails with
`NvVFX_Load failed: ... (code -12)` (`NVCV_ERR_INITIALIZATION`). Upstream issue #4 (same
RTX Pro 6000 / Blackwell hardware) is unresolved.

**Diagnosis path:**
- Reproduced standalone (no torch, `PYTORCH_CUDA_ALLOC_CONF=` unset) so it was clearly an
  NGX init failure, not a torch/WSL artifact.
- `strace -f -e trace=openat,ioctl` on the failing load showed, right before the failure,
  the NGX runtime trying to load **`libnvidia-ngx.so.1`** from `/usr/lib`,
  `/usr/lib/x86_64-linux-gnu`, etc. — all **ENOENT**.

**Root cause:** The NVIDIA Container Toolkit mounts most WSL driver libs into the container
(`libcuda`, `libnvidia-ml`, `libnvdxgdmal`, …) **but not `libnvidia-ngx.so.1`**, which the
RTX VSR NGX backend (`libnvngxruntime.so` → `libnvidia-ngx-vsr.so`) needs. The host has it
at `/usr/lib/wsl/lib/libnvidia-ngx.so.1`; the container does not.

> The user's host `echo /usr/lib/wsl/lib > /etc/ld.so.conf.d/... && ldconfig` did **not**
> help, because that affects the WSL host's loader, not the container's filesystem.

**Fix — vendor the lib and bake it into the image** (NOT a single-file bind mount, which is
fragile — a missing source path makes Docker create a directory in its place):
1. Copied the host lib into the repo: `vendor/wsl/libnvidia-ngx.so.1`
   (4.4 MB; sha256 `3c8d59dc…`; from host `/usr/lib/wsl/lib/libnvidia-ngx.so.1`).
2. `Dockerfile` (near the end, before the entrypoint COPY):
   ```dockerfile
   COPY vendor/wsl/libnvidia-ngx.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1
   RUN ldconfig
   ```
Verified the mechanism live: `docker cp`-ing the lib to that path + `ldconfig` in the
running container made the standalone VSR `load()` succeed ("NGX initialized!"). The
running container therefore already works (the next VSR node run dlopens the copied lib);
the Dockerfile change makes it permanent on the next `docker compose build`.

**Caveats:**
- `libnvidia-ngx.so.1` is **tied to the host NVIDIA driver version** (currently 595.x). If
  the driver updates and VSR breaks again, re-vendor the file from the host.
- It's an NVIDIA-proprietary binary now tracked in git (the `.gitignore` `*.so` rule does
  not match `*.so.1`). Fine for a private repo; reconsider before publishing.
- Belongs in the image, not a volume — no volume-shadowing concerns.

---

## Outstanding — other missing nodes in `crk_ltx2.3-10eros.json`

After the 10S swap, the workflow still references nodes from **other** packs that are
absent or outdated in the current install. Not yet resolved:

| Missing node type | Likely source | Notes |
|---|---|---|
| `LTXVAudioVAEDecode`, `LTXVAudioVAEEncode`, `LTXVConcatAVLatent`, `LTXVSeparateAVLatent`, `LTXVEmptyLatentAudio`, `LTXVConditioning` | ComfyUI-LTXVideo (newer, LTX-2 audio) | Not found in the installed (patched) ComfyUI-LTXVideo — likely needs a newer commit/version of that pack |
| `LTX2LoraLoaderAdvanced`, `LTXVImgToVideoInplaceKJ` | comfyui-kjnodes | Class names exist in installed kjnodes files but workflow flags them missing — may need a kjnodes update or the pack failed to register them |
| `Audio Duration (mtb)` | comfy-mtb | Separate pack |
| `LatentUpscaleModelLoader` | unknown | Not found in any installed pack |

Diagnosis approach: same script as Issue 2, plus a cross-pack grep —
`docker compose exec -T comfyui bash -lc 'grep -rl "<NodeName>" /app/custom_nodes/*/`'`.

## General playbook

- **After rebuilding the image (to update ComfyUI):** re-sync packages —
  `docker compose exec comfyui pip install -r /app/requirements.txt` — then restart.
- **Clean reset of the package volume:**
  `docker compose down && docker volume rm ai-comfyui_site_packages && docker compose up -d`.
  Custom-node packages reinstall automatically on first boot via `entrypoint.sh`
  (slower first start). With `PIP_EXTRA_INDEX_URL` set, `nvidia-vfx` resolves too.
- **Diagnosing a custom node:** check the import-times block in
  `docker compose logs comfyui` — a node printed with `(IMPORT FAILED)` is broken;
  one printed without it loaded fine.
