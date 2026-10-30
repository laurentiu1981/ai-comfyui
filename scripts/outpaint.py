#!/usr/bin/env python3
"""Outpaint every photo in a folder to a target aspect ratio with Qwen Image 2.1.

Driven by scripts/outpaint.sh; run `scripts/outpaint.sh --help` for the options.

How it works, per image:
  1. Read the size (EXIF orientation applied, like ComfyUI's LoadImage does).
  2. If the aspect ratio is already within the tolerance, skip the image.
  3. Otherwise keep one axis at its original size and extend the other one
     to reach the target ratio (never downscale the original).
  4. Build an API-format ComfyUI graph and submit it to the running server:
     - the Qwen3-VL text encoder captions the photo (TextGenerate), so the
       prompt describes this scene (subject, background, lighting);
     - --method lora (default): the gray-padded canvas is reference image_1 of
       TextEncodeQwenImage21 (resolution 0), the outpaint LoRA
       (ausboss/Qwen-Image-2.1-Outpaint-LoRA) fills the gray, prompt = its
       trigger instruction + "Scene: <caption>", no mask, sampling on the
       encoder's own latent;
     - --method controlnet: the Qwen Image 2.1 Fun ControlNet Union runs in
       inpaint mode through the repo's QwenImage21FunInpaint node
       (custom_nodes/comfyui-outpaint-helpers): the gray-padded canvas + pad
       mask go in as keep-mask / masked-image latents, no structural control
       image; the unpadded original is reference image_1;
     - KSampler on the canvas-sized latent -> decode -> resize to the final
       canvas -> paste the untouched original pixels back (ImageCompositeMasked)
       -> SaveImage.
     Measured on two 1-4 MP photos (RTX PRO 6000): lora --full ~18-46 s/image,
     lora (8-step LoRA, default) ~12-26 s, --no-caption saves another ~7 s;
     controlnet --full ~30-80 s, controlnet ~25-35 s. Seam contrast at the
     paste boundary is lowest with lora.
     Things that did NOT work with Qwen Image 2.1 and were dropped: gray-padded
     reference + latent noise mask (padding stays gray / seams), unpadded
     reference + full re-render (flat colour bands), and feeding the padded
     canvas as the ControlNet's structural `image` (flat colour bands again).
     Canvases above --max-megapixels run through the model at a reduced size
     (aspect preserved) and are upscaled afterwards; the original pixels are
     always composited back at full resolution.
  5. Wait for the result, log it, continue with the next image on failure.
"""

import argparse
import base64
import datetime as dt
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("error: Pillow is required on the host (pip install pillow)")

REPO_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_DIR / "input"
OUTPUT_DIR = REPO_DIR / "output"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_OUTPUT = "outpainting/{date}"
DEFAULT_MODEL = "qwen_image_2.1_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"
FAST_LORA = "qwen-image-2.1/p_qwen_image_2.1_8step_v0.1.safetensors"
DEFAULT_CONTROLNET = "qwen_image_2.1_fun_controlnet_union_bf16.safetensors"
DEFAULT_OUTPAINT_LORA = "qwen-image-2.1/qwen-image-2.1-outpaint-v2.safetensors"
# Trigger instruction of ausboss/Qwen-Image-2.1-Outpaint-LoRA; "Scene: <caption>" may follow.
LORA_PROMPT = ("Outpaint the image: replace the solid gray areas with a seamless continuation "
               "of the scene, keeping the existing picture unchanged.")
CAPTION_INSTRUCTION = (
    "Describe this photo in one detailed paragraph of at most 150 words for an image "
    "generator: the subject and pose, clothing, the environment and background, lighting, "
    "colors, camera framing and photographic style. Plain description only, no instructions."
)

PLACEHOLDERS_HELP = """\
Output path placeholders (expanded once, when the script starts):
  {date}      YYYY-MM-DD            {datetime}  YYYY-MM-DD_HH-mm-ss
  {time}      HH-mm-ss              {year} {month} {day}
  {folder}    input folder name     {aspect}    aspect ratio, e.g. 1x1, 16x9, 1.5
"""


# --------------------------------------------------------------------------- args
class Aspect:
    """Target aspect ratio parsed from 'W:H', 'W/H', 'WxH' or a decimal."""

    def __init__(self, text):
        self.text = text.strip()
        s = self.text.lower()
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)", s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if a <= 0 or b <= 0:
                raise argparse.ArgumentTypeError(f"invalid aspect ratio: {text}")
            self.ratio = a / b
            return
        try:
            self.ratio = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid aspect ratio: {text} (use W:H, W/H, WxH or a decimal like 1.5)")
        if self.ratio <= 0:
            raise argparse.ArgumentTypeError(f"invalid aspect ratio: {text}")

    @property
    def slug(self):
        """'16:9' -> '16x9', '1.5' -> '1.5' (safe for folder names)."""
        return re.sub(r"[^0-9a-z.]+", "x", self.text.lower()).strip("x")


def positive_float(text):
    v = float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="scripts/outpaint.sh",
        description="Outpaint all photos in a folder to an aspect ratio with Qwen Image 2.1, "
                    "via the running ComfyUI server.",
        epilog=PLACEHOLDERS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="folder with images (png/jpg/jpeg/webp/bmp); must be under ./input")
    p.add_argument("-a", "--aspect-ratio", required=True, type=Aspect, metavar="RATIO",
                   help="target width:height, e.g. 1:1, 16:9, 4/5, 3x2 or a decimal like 1.5")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                   help="relative = subfolder of ./output used by the workflow's SaveImage; "
                        "absolute or ~ path = results are saved to the default location and "
                        f"then copied there (default: {DEFAULT_OUTPUT})")
    p.add_argument("--tolerance", type=positive_float, default=1.0, metavar="PCT",
                   help="skip images whose ratio is within this percentage of the target (default: 1)")
    p.add_argument("--anchor", choices=["center", "start", "end"], default="center",
                   help="where the original sits on the extended axis: center, start (left/top) "
                        "or end (right/bottom) (default: center)")
    p.add_argument("--max-megapixels", type=positive_float, default=4.0, metavar="MP",
                   help="canvas size the model works at; larger canvases are generated smaller "
                        "and upscaled, the original pixels are always kept at full resolution "
                        "(default: 4 = native 2K)")
    p.add_argument("-m", "--method", choices=["lora", "controlnet"], default="lora",
                   help="lora (default): gray-padded canvas as reference + the outpaint LoRA, no mask "
                        "(ausboss/Qwen-Image-2.1-Outpaint-LoRA recipe; faster, cleaner seam). "
                        "controlnet: Fun ControlNet Union inpaint on the gray-padded canvas, original as reference")
    p.add_argument("--prompt", metavar="TEXT",
                   help="fixed scene description for every image instead of the automatic "
                        "per-image caption; {sides} expands to e.g. 'left and right'")
    p.add_argument("--no-caption", action="store_true",
                   help="skip the per-image caption (controlnet: empty prompt; lora: trigger instruction only)")
    p.add_argument("--show-prompt", action="store_true", help="print the caption used for each image")
    p.add_argument("--no-reference", action="store_true",
                   help="controlnet method: do not pass the original as reference image_1")
    p.add_argument("--controlnet", default=DEFAULT_CONTROLNET, metavar="NAME",
                   help=f"Fun ControlNet Union patch under models/model_patches (default: {DEFAULT_CONTROLNET})")
    p.add_argument("--controlnet-strength", type=float, default=1.0, metavar="X",
                   help="ControlNet inpaint strength (default: 1.0)")
    p.add_argument("--outpaint-lora", default=DEFAULT_OUTPAINT_LORA, metavar="NAME",
                   help=f"lora method: outpaint LoRA under models/loras (default: {DEFAULT_OUTPAINT_LORA})")
    p.add_argument("--outpaint-lora-strength", type=float, default=1.0, metavar="X",
                   help="lora method: outpaint LoRA strength (default: 1.0)")
    p.add_argument("--negative", default="", help="negative prompt (only used when --cfg > 1)")
    p.add_argument("--steps", type=int, help="sampler steps (default: 8, or 25 with --full)")
    p.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default: 1, the official path)")
    p.add_argument("--sampler", default="euler", help="sampler name (default: euler)")
    p.add_argument("--scheduler", default="simple", help="scheduler (default: simple)")
    p.add_argument("--seed", type=int, help="fixed seed for every image (default: random per image)")
    p.add_argument("--full", action="store_true",
                   help=f"full-quality sampling: 25 steps without the 8-step LoRA ({FAST_LORA}, on by default)")
    p.add_argument("--lora", metavar="NAME", help="extra model-only LoRA to apply (path under models/loras)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"diffusion model (default: {DEFAULT_MODEL})")
    p.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER,
                   help=f"text encoder (default: {DEFAULT_TEXT_ENCODER})")
    p.add_argument("--vae", default=DEFAULT_VAE, help=f"VAE (default: {DEFAULT_VAE})")
    p.add_argument("--limit", type=int, metavar="N", help="only process the first N images")
    p.add_argument("--timeout", type=int, default=900, metavar="SEC",
                   help="max wait per image, including model loading (default: 900)")
    p.add_argument("--url", default=os.environ.get("COMFYUI_URL", "http://localhost:8188"),
                   help="ComfyUI server (default: $COMFYUI_URL or http://localhost:8188)")
    p.add_argument("--dry-run", action="store_true", help="print the plan for every image, submit nothing")
    args = p.parse_args(argv)
    if args.full:
        args.steps = args.steps or 25
    else:
        args.lora = args.lora or FAST_LORA
        args.steps = args.steps or 8
    return args


# --------------------------------------------------------------------------- helpers
def load_dotenv(path):
    """Read KEY=VALUE lines from the repo .env without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def expand_placeholders(template, ctx):
    def repl(m):
        key = m.group(1)
        if key not in ctx:
            sys.exit(f"error: unknown placeholder {{{key}}} in output path; "
                     f"supported: {', '.join('{' + k + '}' for k in ctx)}")
        return ctx[key]
    return re.sub(r"\{(\w+)\}", repl, template)


def image_size(path):
    """(width, height) after EXIF orientation, matching ComfyUI's LoadImage."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(0x0112)
        except Exception:
            orientation = None
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return w, h


def split_pad(total, anchor):
    if anchor == "start":
        return 0, total
    if anchor == "end":
        return total, 0
    return total // 2, total - total // 2


def plan_image(w, h, ratio, tolerance_pct, max_pixels, anchor):
    """Return None when the ratio already matches, else the canvas plan dict."""
    if abs((w / h) / ratio - 1.0) * 100.0 <= tolerance_pct:
        return None
    if w / h < ratio:            # too narrow -> extend width
        W, H, axis = max(w, round(h * ratio)), h, "width"
    else:                        # too wide -> extend height
        W, H, axis = w, max(h, round(w / ratio)), "height"

    # Model canvas: same aspect, multiples of 32, at most max_pixels.
    scale = min(1.0, math.sqrt(max_pixels / (W * H)))
    MW = max(32, round(W * scale / 32) * 32)
    MH = max(32, round(H * scale / 32) * 32)
    mw = min(MW, max(1, round(w * MW / W)))
    mh = min(MH, max(1, round(h * MH / H)))

    def pads(cw, ch, iw, ih):
        pl, pr = split_pad(cw - iw, anchor if axis == "width" else "center")
        pt, pb = split_pad(ch - ih, anchor if axis == "height" else "center")
        return pl, pt, pr, pb

    sides = []
    fl, ft, fr, fb = pads(W, H, w, h)
    for name, v in (("left", fl), ("top", ft), ("right", fr), ("bottom", fb)):
        if v > 0:
            sides.append(name)
    return {
        "W": W, "H": H, "axis": axis, "final_pads": (fl, ft, fr, fb),
        "MW": MW, "MH": MH, "mw": mw, "mh": mh, "model_pads": pads(MW, MH, mw, mh),
        "scale": MW / W,
        "sides": " and ".join(sides) if len(sides) <= 2 else ", ".join(sides[:-1]) + " and " + sides[-1],
    }


def build_graph(plan, image_ref, args, seed, filename_prefix):
    g = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": args.model, "weight_dtype": "default"}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": args.text_encoder, "type": "qwen_image", "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}}
    model = ["1", 0]
    if args.method == "lora":
        g["40"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": args.outpaint_lora,
                              "strength_model": args.outpaint_lora_strength}}
        model = ["40", 0]
    if args.lora:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": args.lora, "strength_model": 1.0}}
        model = ["4", 0]
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": image_ref}}
    g["6"] = {"class_type": "ImageScale",
              "inputs": {"image": ["5", 0], "upscale_method": "lanczos",
                         "width": plan["mw"], "height": plan["mh"], "crop": "disabled"}}
    # Gray-padded canvas: known pixels + pad mask for the ControlNet inpaint channels,
    # and the latent size for sampling.
    ml, mt, mr, mb = plan["model_pads"]
    g["7"] = {"class_type": "ImagePadForOutpaint",
              "inputs": {"image": ["6", 0], "left": ml, "top": mt, "right": mr, "bottom": mb,
                         "feathering": 0}}
    if args.method == "controlnet":
        g["30"] = {"class_type": "ModelPatchLoader", "inputs": {"name": args.controlnet}}
        g["31"] = {"class_type": "QwenImage21FunInpaint",
                   "inputs": {"model": model, "model_patch": ["30", 0], "vae": ["3", 0],
                              "inpaint_image": ["7", 0], "mask": ["7", 1],
                              "strength": args.controlnet_strength}}
        model = ["31", 0]

    # Scene description: fixed --prompt, else a caption written by the Qwen3-VL text
    # encoder (TextGenerate), else nothing.
    scene = None
    if args.prompt:
        scene = args.prompt.replace("{sides}", plan["sides"])
    elif not args.no_caption:
        g["21"] = {"class_type": "TextGenerate",
                   "inputs": {"clip": ["2", 0], "image": ["5", 0], "prompt": CAPTION_INSTRUCTION,
                              "max_length": 512, "sampling_mode": "on",
                              "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 20,
                              "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                              "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": seed,
                              "thinking": False, "use_default_template": True, "mtp": "auto"}}
        scene = ["21", 0]
    if args.method == "lora":
        if scene is None:
            prompt = LORA_PROMPT
        else:
            g["24"] = {"class_type": "StringConcatenate",
                       "inputs": {"string_a": LORA_PROMPT, "string_b": scene, "delimiter": " Scene: "}}
            prompt = ["24", 0]
    else:
        prompt = scene if scene is not None else ""
    if not isinstance(prompt, str):
        g["23"] = {"class_type": "PreviewAny", "inputs": {"source": prompt}}

    enc = {"clip": ["2", 0], "vae": ["3", 0], "resolution": 0,
           "prompt": prompt, "negative_prompt": args.negative}
    if args.method == "lora":
        enc["images.image_1"] = ["7", 0]       # the gray-padded canvas is the reference
    elif not args.no_reference:
        enc["images.image_1"] = ["6", 0]
    g["8"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    if args.method == "lora":
        latent = ["8", 2]                      # empty latent at the reference (canvas) size
    else:
        g["9"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["3", 0]}}
        latent = ["9", 0]
    g["11"] = {"class_type": "KSampler",
               "inputs": {"model": model, "seed": seed, "steps": args.steps, "cfg": args.cfg,
                          "sampler_name": args.sampler, "scheduler": args.scheduler,
                          "positive": ["8", 0], "negative": ["8", 1],
                          "latent_image": latent, "denoise": 1.0}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    g["13"] = {"class_type": "ImageScale",
               "inputs": {"image": ["12", 0], "upscale_method": "lanczos",
                          "width": plan["W"], "height": plan["H"], "crop": "disabled"}}
    fl, ft, _, _ = plan["final_pads"]
    g["14"] = {"class_type": "ImageCompositeMasked",
               "inputs": {"destination": ["13", 0], "source": ["5", 0],
                          "x": fl, "y": ft, "resize_source": False}}
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": filename_prefix}}
    return g


# --------------------------------------------------------------------------- comfyui client
class ComfyError(Exception):
    pass


class Comfy:
    def __init__(self, url, user, password):
        self.url = url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if user or password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"
        self.client_id = f"outpaint-{os.getpid()}-{int(time.time())}"

    def _request(self, path, data=None, timeout=30):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.url + path, data=body, headers=self.headers,
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                j = json.loads(detail)
                msgs = [j.get("error", {}).get("message", "")]
                for nid, ne in (j.get("node_errors") or {}).items():
                    cls = ne.get("class_type", "?")
                    for err in ne.get("errors", []):
                        msgs.append(f"node {nid} ({cls}): {err.get('message')} {err.get('details', '')}".strip())
                detail = "; ".join(m for m in msgs if m) or detail
            except ValueError:
                pass
            raise ComfyError(f"HTTP {e.code} on {path}: {detail[:800]}")
        except urllib.error.URLError as e:
            raise ComfyError(f"cannot reach {self.url}: {e.reason}")
        return json.loads(raw) if raw else {}

    def check(self):
        return self._request("/system_stats")

    def submit(self, graph):
        r = self._request("/prompt", {"prompt": graph, "client_id": self.client_id})
        return r["prompt_id"]

    def interrupt(self):
        try:
            self._request("/interrupt", {})
        except ComfyError:
            pass

    def wait(self, prompt_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self._request(f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    detail = "execution error"
                    for m in status.get("messages", []):
                        if m[0] == "execution_error":
                            d = m[1]
                            detail = f"{d.get('node_type')}: {d.get('exception_message')}"
                    raise ComfyError(detail)
                files, texts = [], []
                for out in entry.get("outputs", {}).values():
                    for img in out.get("images", []):
                        if img.get("type", "output") == "output":
                            files.append(Path(img.get("subfolder", "")) / img["filename"])
                    texts.extend(t for t in out.get("text", []) if isinstance(t, str))
                return files, texts
            time.sleep(1.0)
        self.interrupt()
        raise ComfyError(f"timed out after {timeout}s")


# --------------------------------------------------------------------------- main
def main(argv):
    args = parse_args(argv)
    load_dotenv(REPO_DIR / ".env")
    started = dt.datetime.now()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        sys.exit(f"error: '{args.folder}' is not a directory")
    folder = folder.resolve()
    try:
        rel = folder.relative_to(INPUT_DIR.resolve())
    except ValueError:
        sys.exit(f"error: folder must be inside {INPUT_DIR} (the container only sees ./input)")

    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"error: no images (png/jpg/jpeg/webp/bmp) in {folder}")
    if args.limit:
        images = images[:args.limit]

    ctx = {
        "date": started.strftime("%Y-%m-%d"),
        "datetime": started.strftime("%Y-%m-%d_%H-%M-%S"),
        "time": started.strftime("%H-%M-%S"),
        "year": started.strftime("%Y"), "month": started.strftime("%m"), "day": started.strftime("%d"),
        "folder": folder.name,
        "aspect": args.aspect_ratio.slug,
    }
    out = expand_placeholders(args.output, ctx)
    host_dest = None
    if out.startswith(("/", "~")):
        dest = Path(out).expanduser().resolve()
        try:
            prefix_dir = dest.relative_to(OUTPUT_DIR.resolve()).as_posix()
        except ValueError:
            prefix_dir = expand_placeholders(DEFAULT_OUTPUT, ctx)
            host_dest = dest
    else:
        prefix_dir = out.strip("/")
    if not prefix_dir or ".." in Path(prefix_dir).parts:
        sys.exit(f"error: invalid output path '{out}'")

    max_pixels = args.max_megapixels * 1024 * 1024
    print(f"Source      : {len(images)} image(s) in input/{rel.as_posix()}")
    print(f"Aspect      : {args.aspect_ratio.text} ({args.aspect_ratio.ratio:.4f}), tolerance {args.tolerance:g}%, anchor {args.anchor}")
    print(f"Output      : output/{prefix_dir}" + (f"  ->  copied to {host_dest}" if host_dest else ""))
    if args.method == "controlnet":
        print(f"Method      : controlnet | {args.model} + {args.controlnet} (inpaint, strength {args.controlnet_strength:g})"
              + (" | no reference" if args.no_reference else ""))
    else:
        print(f"Method      : lora | {args.model} + {args.outpaint_lora} (strength {args.outpaint_lora_strength:g})")
    print(f"Sampling    : steps={args.steps} cfg={args.cfg:g} {args.sampler}/{args.scheduler}"
          + (f" | lora={args.lora}" if args.lora else "") + f" | max {args.max_megapixels:g} MP"
          + (" | fixed prompt" if args.prompt else " | no caption" if args.no_caption else " | auto caption"))

    comfy = Comfy(args.url, os.environ.get("COMFYUI_USERNAME", "admin"),
                  os.environ.get("COMFYUI_PASSWORD", "password"))
    if not args.dry_run:
        try:
            comfy.check()
        except ComfyError as e:
            sys.exit(f"error: {e}")
        if host_dest:
            host_dest.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = [], [], []
    width = len(str(len(images)))
    for i, path in enumerate(images, 1):
        tag = f"[{i:{width}}/{len(images)}] {path.name}"
        try:
            w, h = image_size(path)
        except Exception as e:
            print(f"{tag}: FAILED to read image ({e})")
            failed.append(path.name)
            continue
        plan = plan_image(w, h, args.aspect_ratio.ratio, args.tolerance, max_pixels, args.anchor)
        if plan is None:
            print(f"{tag}: {w}x{h} already {args.aspect_ratio.text}, skipped")
            skipped.append(path.name)
            continue
        fl, ft, fr, fb = plan["final_pads"]
        model_note = (f", model canvas {plan['MW']}x{plan['MH']}"
                      if (plan["MW"], plan["MH"]) != (plan["W"], plan["H"]) else "")
        print(f"{tag}: {w}x{h} -> {plan['W']}x{plan['H']} "
              f"(pad l{fl} t{ft} r{fr} b{fb}{model_note})", end="", flush=True)
        if args.dry_run:
            print()
            done.append(path.name)
            continue

        seed = args.seed if args.seed is not None else random.randrange(0, 2**53)
        graph = build_graph(plan, f"{rel.as_posix()}/{path.name}",
                            args, seed, f"{prefix_dir}/{path.stem}")
        t0 = time.time()
        try:
            prompt_id = comfy.submit(graph)
        except ComfyError as e:
            print(f"\n{tag}: FAILED to submit: {e}")
            failed.append(path.name)
            if not done:
                sys.exit("error: the first workflow was rejected by the server; "
                         "fix the error above before retrying (every image would fail the same way)")
            continue
        try:
            files, texts = comfy.wait(prompt_id, args.timeout)
        except KeyboardInterrupt:
            comfy.interrupt()
            print("\ninterrupted")
            failed.append(path.name)
            break
        except ComfyError as e:
            print(f"\n{tag}: FAILED: {e}")
            failed.append(path.name)
            continue

        names = []
        for f in files:
            src = OUTPUT_DIR / f
            if host_dest:
                try:
                    shutil.copy2(src, host_dest / f.name)
                except OSError as e:
                    print(f"\n{tag}: could not copy {src} -> {host_dest}: {e}")
            names.append(f.name)
        print(f" -> {', '.join(names) or '(no output files)'} [{time.time() - t0:.0f}s, seed {seed}]")
        if args.show_prompt and texts:
            print("    prompt: " + " ".join(texts).strip().replace("\n", " "))
        done.append(path.name)

    print()
    verb = "planned" if args.dry_run else "outpainted"
    print(f"Summary     : {len(done)} {verb}, {len(skipped)} skipped (ratio already matched), {len(failed)} failed")
    if failed:
        print("Failed      : " + ", ".join(failed))
    if not args.dry_run and done:
        print(f"Results     : {host_dest if host_dest else OUTPUT_DIR / prefix_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
