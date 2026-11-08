#!/usr/bin/env python3
"""Dress the person in one photo, or in every photo of a folder, in the outfit from
another picture with Qwen Image 2.1 and the ausboss Outfit Swap Consistency LoRA.

Driven by scripts/outfit.sh; run `scripts/outfit.sh --help` for the options.

This is the API-format equivalent of ausboss' ComfyUI workflow
(Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA/workflows/qwen-image-2.1-outfit-swap.json),
built from core ComfyUI nodes only (the AusBoss resize node is replaced by
ImageScaleToTotalPixels). Per image:
  1. The model works at the person picture's own size rounded to multiples of 32,
     downscaled only when it exceeds --max-megapixels (default 4, Qwen's native 2K);
     a source whose sides are multiples of 32 and within the cap is never resampled.
     --megapixels forces a fixed working size instead (the author used 2). The outfit
     picture is a reference only and is scaled to about --outfit-megapixels (default 1).
  2. The request is "Dress <who> in image 1 in <outfit phrase>. Keep their face, hair,
     hands, pose and the background exactly the same. <extra>" where <who> defaults to
     "the person", <extra> to "Replace everything they wear." and <outfit phrase> is:
       - --prompt TEXT        your own phrase, e.g. "the red leather jacket shown in image 2";
       - --no-caption         "the outfit shown in image 2";
       - default              written by the Qwen3-VL text encoder (TextGenerate) from the
                              outfit picture. As in the original workflow, the encoder first
                              answers whether the person picture shows legs and feet, and the
                              describer is told to leave out shoes (no feet) or everything
                              below the waist (no legs) so the picture is not zoomed out to
                              fit the whole outfit; --coverage overrides those two checks.
  3. TextEncodeQwenImage21 (resolution 0) gets the person as image_1, the outfit as
     image_2 and the request; the outfit LoRA (strength 1) and, unless --full, a speed
     LoRA (--speed-lora: 8step, viggle or a file) are applied to the model;
     QwenImage21Cache caches the reference prefix.
  4. KSampler on the encoder's latent (cfg 1, euler/simple), VAEDecode, drop the alpha
     channel, scale back to the source dimensions when the working size differs
     (lanczos; --model-size keeps the working size), SaveImage.
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

DEFAULT_OUTPUT = "outfit/{date}/{outfit}"
DEFAULT_MODEL = "qwen_image_2.1_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = "qwen3vl_8b_int8_convrot.safetensors"
DEFAULT_VAE = "qwen_image_2.1_vae_bf16.safetensors"
SPEED_LORAS = {
    "8step": "qwen-image-2.1/p_qwen_image_2.1_8step_v0.1.safetensors",
    # The one the LoRA author tested at 8 steps (Qwen Research License, research use).
    "viggle": "qwen-image-2.1/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors",
}
FAST_LORA = SPEED_LORAS["8step"]
DEFAULT_OUTFIT_LORA = "qwen-image-2.1/qwen-image-2.1-outfit-swap.safetensors"

DEFAULT_WHO = "the person"
DEFAULT_EXTRA = "Replace everything they wear."
DEFAULT_TEMPLATE = ("Dress {who} in image 1 in {outfit}. Keep their face, hair, hands, pose "
                    "and the background exactly the same. {extra}")
GENERIC_OUTFIT = "the outfit shown in image 2"

# Text-encoder instructions, taken from the original workflow's "Write the prompt" subgraph.
DESCRIBE_HEAD = ("You write short image-edit descriptions. Reply with one phrase only, "
                 "no preamble, no quotes.\n\n")
DESCRIBE_FULL = DESCRIBE_HEAD + (
    "Look at the clothing in this picture. Describe the outfit in one short phrase for an "
    "image editor: name every garment, the top, the bottom (skirt, shorts, trousers, dress) "
    "and any shoes or accessories, each with its color, pattern and the details that identify "
    "it (collar, straps, sleeves, trim, buttons, belt, chain). Start with 'the' and end with "
    "'shown in image 2'. Do not describe the background or any person.")
DESCRIBE_NO_SHOES = DESCRIBE_HEAD + (
    "Look at the clothing in this picture. Describe the outfit in one short phrase for an "
    "image editor: name every garment, the top, the bottom (skirt, shorts, trousers, dress) "
    "and any accessories, each with its color, pattern and the details that identify it "
    "(collar, straps, sleeves, trim, buttons, belt, chain). Leave out shoes, boots and socks: "
    "the picture the outfit goes on does not show the feet. Start with 'the' and end with "
    "'shown in image 2, without the shoes'. Do not describe the background or any person.")
DESCRIBE_TOP = DESCRIBE_HEAD + (
    "Look at the clothing in this picture. The picture it goes on shows the person only from "
    "the chest or waist up, so describe only what is worn on the upper body, in one short "
    "phrase for an image editor: the outer layer (jacket, coat, cardigan, shirt, sweater, or "
    "the top of a dress or overalls) and what shows under it, each with its color, pattern and "
    "the details that identify it (collar, sleeves, buttons, trim, straps). Leave out trousers, "
    "skirts, shorts, the lower part of a dress, shoes and socks. Start with 'the' and end with "
    "'shown in image 2'. Do not describe the background or any person.")
FEET_QUESTION = "Are the person's feet or shoes in this picture? Answer yes or no."
LEGS_QUESTION = "Are the person's legs in this picture? Answer yes or no."
YES_PATTERN = r"^\s*yes\b"
TRIM_PATTERN = r"^[\s\"']+|[\s\"'.]+$"

PLACEHOLDERS_HELP = """\
Output path placeholders (expanded once, when the script starts):
  {date}      YYYY-MM-DD            {datetime}  YYYY-MM-DD_HH-mm-ss
  {time}      HH-mm-ss              {year} {month} {day}
  {folder}    input folder name     {outfit}    outfit file name without extension
"""


# --------------------------------------------------------------------------- args
def positive_float(text):
    v = float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="scripts/outfit.sh",
        description="Dress the person in a photo (or in every photo of a folder) in the outfit "
                    "from another picture, with Qwen Image 2.1 + the Outfit Swap LoRA, via the "
                    "running ComfyUI server.",
        epilog=PLACEHOLDERS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="image file or folder with images (png/jpg/jpeg/webp/bmp); must be under ./input")
    p.add_argument("--outfit", required=True, metavar="IMAGE",
                   help="picture of the outfit (a person wearing it, a flat lay or a product shot); "
                        "must be under ./input")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                   help="relative = subfolder of ./output used by the workflow's SaveImage; "
                        "absolute or ~ path = results are saved to the default location and "
                        f"then copied there (default: {DEFAULT_OUTPUT})")
    p.add_argument("--who", default=DEFAULT_WHO, metavar="TEXT",
                   help="who to dress, for group photos: 'the woman on the left', 'the man with the "
                        f"beard' (default: '{DEFAULT_WHO}')")
    p.add_argument("--prompt", metavar="TEXT",
                   help="fixed description of the outfit instead of the automatic one, e.g. "
                        "'the red leather jacket shown in image 2'")
    p.add_argument("--no-caption", action="store_true",
                   help=f"skip the automatic outfit description and ask for '{GENERIC_OUTFIT}'")
    p.add_argument("--coverage", choices=["auto", "full", "no-shoes", "top"], default="auto",
                   help="what the describer is told to cover: auto (default) asks the text encoder "
                        "whether the person picture shows legs and feet; full = whole outfit incl. "
                        "shoes; no-shoes = whole outfit without shoes; top = upper body only")
    p.add_argument("--extra", default=DEFAULT_EXTRA, metavar="TEXT",
                   help="sentence added at the end of the request; use 'Keep their own shoes and "
                        f"legwear.' to keep them, or '' for nothing (default: '{DEFAULT_EXTRA}')")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, metavar="TEXT",
                   help="the whole request, with {who}, {outfit} and {extra} placeholders "
                        f"(default: '{DEFAULT_TEMPLATE}')")
    p.add_argument("--show-prompt", action="store_true", help="print the request sent to Qwen for each image")
    p.add_argument("--max-megapixels", type=positive_float, default=4.0, metavar="MP",
                   help="the model works at the source size (multiples of 32); larger sources are "
                        "downscaled to this many megapixels and the result upscaled back "
                        "(default: 4 = Qwen's native 2K; the LoRA author tested at 2)")
    p.add_argument("--megapixels", type=positive_float, metavar="MP",
                   help="force a fixed working size instead of the source size (the author's workflow uses 2)")
    p.add_argument("--outfit-megapixels", type=positive_float, default=1.0, metavar="MP",
                   help="size the outfit picture is scaled to (default: 1)")
    p.add_argument("--model-size", action="store_true",
                   help="keep the result at the model's working size instead of scaling it back to "
                        "the source dimensions when they differ")
    p.add_argument("--outfit-lora", default=DEFAULT_OUTFIT_LORA, metavar="NAME",
                   help=f"outfit swap LoRA under models/loras (default: {DEFAULT_OUTFIT_LORA})")
    p.add_argument("--outfit-lora-strength", type=float, default=1.0, metavar="X",
                   help="outfit swap LoRA strength; 0 disables it (default: 1.0, the author's recommendation)")
    p.add_argument("--no-outfit-lora", action="store_true", help="plain Qwen Image 2.1 edit, without the outfit swap LoRA")
    p.add_argument("--no-cache", action="store_true", help="do not add the QwenImage21Cache node")
    p.add_argument("--negative", default="", help="negative prompt (only used when --cfg > 1)")
    p.add_argument("--steps", type=int, help="sampler steps (default: 8, or 25 with --full)")
    p.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default: 1, the official path)")
    p.add_argument("--sampler", default="euler", help="sampler name (default: euler)")
    p.add_argument("--scheduler", default="simple", help="scheduler (default: simple)")
    p.add_argument("--seed", type=int, help="fixed seed for every image (default: random per image)")
    p.add_argument("--full", action="store_true",
                   help="full-quality sampling: 25 steps without the speed LoRA (the author's tested recipe)")
    p.add_argument("--speed-lora", default="8step", metavar="NAME",
                   help="speed LoRA for 8-step sampling: '8step' (default, "
                        f"{SPEED_LORAS['8step']}), 'viggle' (Viggle Turbo, {SPEED_LORAS['viggle']}, "
                        "the one the outfit LoRA author tested with) or a file under models/loras")
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
        args.lora = args.lora or SPEED_LORAS.get(args.speed_lora, args.speed_lora)
        args.steps = args.steps or 8
    if args.no_outfit_lora:
        args.outfit_lora_strength = 0.0
    for key in re.findall(r"\{(\w+)\}", args.template):
        if key not in ("who", "outfit", "extra"):
            p.error(f"unknown placeholder {{{key}}} in --template (use {{who}}, {{outfit}}, {{extra}})")
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


def input_relative(path, what):
    """Resolve a path and return it relative to ./input, or exit."""
    p = Path(path).expanduser()
    if not p.exists():
        sys.exit(f"error: {what} '{path}' does not exist")
    p = p.resolve()
    try:
        return p, p.relative_to(INPUT_DIR.resolve())
    except ValueError:
        sys.exit(f"error: {what} must be inside {INPUT_DIR} (the container only sees ./input)")


def working_size(w, h, args):
    """Model working size: the source size at multiples of 32, capped at --max-megapixels,
    or a fixed --megapixels. Equal to (w, h) when nothing needs to change."""
    if args.megapixels:
        scale = math.sqrt(args.megapixels * 1024 * 1024 / (w * h))
    else:
        scale = min(1.0, math.sqrt(args.max_megapixels * 1024 * 1024 / (w * h)))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


def build_graph(person_ref, outfit_ref, args, seed, filename_prefix, size, model_size):
    g = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": args.model, "weight_dtype": "default"}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": args.text_encoder, "type": "qwen_image", "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}}
    model = ["1", 0]
    if args.outfit_lora_strength:
        g["40"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": args.outfit_lora,
                              "strength_model": args.outfit_lora_strength}}
        model = ["40", 0]
    if args.lora:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": model, "lora_name": args.lora, "strength_model": 1.0}}
        model = ["4", 0]
    if not args.no_cache:
        g["41"] = {"class_type": "QwenImage21Cache",
                   "inputs": {"model": model, "device": "auto", "dtype": "default"}}
        model = ["41", 0]

    g["5"] = {"class_type": "LoadImage", "inputs": {"image": person_ref}}
    g["6"] = {"class_type": "LoadImage", "inputs": {"image": outfit_ref}}
    person = ["5", 0]
    if model_size != size:
        g["7"] = {"class_type": "ImageScale",
                  "inputs": {"image": person, "upscale_method": "lanczos",
                             "width": model_size[0], "height": model_size[1], "crop": "disabled"}}
        person = ["7", 0]
    g["8"] = {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                         "megapixels": args.outfit_megapixels, "resolution_steps": 32}}

    def generate(nid, image, prompt, max_length):
        g[nid] = {"class_type": "TextGenerate",
                  "inputs": {"clip": ["2", 0], "image": image, "prompt": prompt,
                             "max_length": max_length, "sampling_mode": "off",
                             "thinking": True, "use_default_template": True, "mtp": "auto"}}
        return [nid, 0]

    # The outfit phrase: fixed, generic, or described by the text encoder.
    if args.prompt:
        outfit = args.prompt
    elif args.no_caption:
        outfit = GENERIC_OUTFIT
    else:
        g["61"] = {"class_type": "ImageScaleToTotalPixels",
                   "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                              "megapixels": 0.6, "resolution_steps": 32}}
        if args.coverage == "auto":
            g["60"] = {"class_type": "ImageScaleToTotalPixels",
                       "inputs": {"image": ["5", 0], "upscale_method": "lanczos",
                                  "megapixels": 0.6, "resolution_steps": 32}}
            feet = generate("62", ["60", 0], FEET_QUESTION, 16)
            g["63"] = {"class_type": "RegexMatch",
                       "inputs": {"string": feet, "regex_pattern": YES_PATTERN,
                                  "case_insensitive": True, "multiline": False, "dotall": False}}
            legs = generate("64", ["60", 0], LEGS_QUESTION, 16)
            g["65"] = {"class_type": "RegexMatch",
                       "inputs": {"string": legs, "regex_pattern": YES_PATTERN,
                                  "case_insensitive": True, "multiline": False, "dotall": False}}
            g["66"] = {"class_type": "ComfySwitchNode",
                       "inputs": {"switch": ["63", 0], "on_true": DESCRIBE_FULL, "on_false": DESCRIBE_NO_SHOES}}
            g["67"] = {"class_type": "ComfySwitchNode",
                       "inputs": {"switch": ["65", 0], "on_true": ["66", 0], "on_false": DESCRIBE_TOP}}
            instruction = ["67", 0]
        else:
            instruction = {"full": DESCRIBE_FULL, "no-shoes": DESCRIBE_NO_SHOES, "top": DESCRIBE_TOP}[args.coverage]
        described = generate("68", ["61", 0], instruction, 200)
        g["69"] = {"class_type": "RegexReplace",
                   "inputs": {"string": described, "regex_pattern": TRIM_PATTERN, "replace": "",
                              "case_insensitive": False, "multiline": False, "dotall": False, "count": 0}}
        outfit = ["69", 0]

    if isinstance(outfit, str):
        prompt = args.template.format(who=args.who, outfit=outfit, extra=args.extra).strip()
    else:
        f_string = args.template.format(who="{a}", outfit="{b}", extra="{c}")
        g["24"] = {"class_type": "StringFormat",
                   "inputs": {"f_string": f_string, "values.a": args.who, "values.b": outfit,
                              "values.c": args.extra}}
        prompt = ["24", 0]
        g["23"] = {"class_type": "PreviewAny", "inputs": {"source": prompt}}

    g["9"] = {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "resolution": 0,
                         "prompt": prompt, "negative_prompt": args.negative,
                         "images.image_1": person, "images.image_2": ["8", 0]}}
    g["11"] = {"class_type": "KSampler",
               "inputs": {"model": model, "seed": seed, "steps": args.steps, "cfg": args.cfg,
                          "sampler_name": args.sampler, "scheduler": args.scheduler,
                          "positive": ["9", 0], "negative": ["9", 1],
                          "latent_image": ["9", 2], "denoise": 1.0}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    g["13"] = {"class_type": "SplitImageWithAlpha", "inputs": {"image": ["12", 0]}}
    image = ["13", 0]
    if not args.model_size and model_size != size:
        g["14"] = {"class_type": "ImageScale",
                   "inputs": {"image": image, "upscale_method": "lanczos",
                              "width": size[0], "height": size[1], "crop": "disabled"}}
        image = ["14", 0]
    g["15"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": filename_prefix}}
    return g, prompt


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
        self.client_id = f"outfit-{os.getpid()}-{int(time.time())}"

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

    source, _ = input_relative(args.source, "source")
    if source.is_dir():
        folder = source
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            sys.exit(f"error: no images (png/jpg/jpeg/webp/bmp) in {folder}")
    elif source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        folder, images = source.parent, [source]
    else:
        sys.exit(f"error: '{args.source}' is neither a folder nor an image (png/jpg/jpeg/webp/bmp)")
    rel = folder.resolve().relative_to(INPUT_DIR.resolve())
    if args.limit:
        images = images[:args.limit]

    outfit_path, outfit_rel = input_relative(args.outfit, "--outfit")
    if not outfit_path.is_file() or outfit_path.suffix.lower() not in IMAGE_EXTS:
        sys.exit(f"error: --outfit '{args.outfit}' is not an image (png/jpg/jpeg/webp/bmp)")
    try:
        ow, oh = image_size(outfit_path)
    except Exception as e:
        sys.exit(f"error: cannot read --outfit '{args.outfit}': {e}")

    ctx = {
        "date": started.strftime("%Y-%m-%d"),
        "datetime": started.strftime("%Y-%m-%d_%H-%M-%S"),
        "time": started.strftime("%H-%M-%S"),
        "year": started.strftime("%Y"), "month": started.strftime("%m"), "day": started.strftime("%d"),
        "folder": folder.name,
        "outfit": outfit_path.stem,
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

    if args.prompt:
        outfit_note = f"fixed: '{args.prompt}'"
    elif args.no_caption:
        outfit_note = f"generic: '{GENERIC_OUTFIT}'"
    else:
        outfit_note = f"described by the text encoder (coverage {args.coverage})"
    source_note = (f"{len(images)} image(s) in input/{rel.as_posix()}" if source.is_dir()
                   else f"input/{rel.as_posix()}/{images[0].name}")
    print(f"Source      : {source_note}")
    print(f"Outfit      : input/{outfit_rel.as_posix()} ({ow}x{oh}) -> {outfit_note}")
    print(f"Request     : who='{args.who}' | extra='{args.extra}'")
    print(f"Output      : output/{prefix_dir}" + (f"  ->  copied to {host_dest}" if host_dest else "")
          + (f" | fixed {args.megapixels:g} MP" if args.megapixels else f" | source size, max {args.max_megapixels:g} MP")
          + (" | keep model size" if args.model_size else ""))
    lora_note = (f"{args.outfit_lora} (strength {args.outfit_lora_strength:g})"
                 if args.outfit_lora_strength else "no outfit LoRA")
    print(f"Model       : {args.model} + {lora_note}" + (" | no cache" if args.no_cache else ""))
    print(f"Sampling    : steps={args.steps} cfg={args.cfg:g} {args.sampler}/{args.scheduler}"
          + (f" | lora={args.lora}" if args.lora else ""))

    comfy = Comfy(args.url, os.environ.get("COMFYUI_USERNAME", "admin"),
                  os.environ.get("COMFYUI_PASSWORD", "password"))
    if not args.dry_run:
        try:
            comfy.check()
        except ComfyError as e:
            sys.exit(f"error: {e}")
        if host_dest:
            host_dest.mkdir(parents=True, exist_ok=True)

    done, failed = [], []
    width = len(str(len(images)))
    for i, path in enumerate(images, 1):
        tag = f"[{i:{width}}/{len(images)}] {path.name}"
        try:
            w, h = image_size(path)
        except Exception as e:
            print(f"{tag}: FAILED to read image ({e})")
            failed.append(path.name)
            continue
        mw, mh = working_size(w, h, args)
        model_note = f" (model {mw}x{mh})" if (mw, mh) != (w, h) else ""
        print(f"{tag}: {w}x{h}{model_note}", end="", flush=True)
        if args.dry_run:
            print()
            done.append(path.name)
            continue

        seed = args.seed if args.seed is not None else random.randrange(0, 2**53)
        graph, prompt = build_graph(f"{rel.as_posix()}/{path.name}", outfit_rel.as_posix(),
                                    args, seed, f"{prefix_dir}/{path.stem}", (w, h), (mw, mh))
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
        if args.show_prompt:
            text = prompt if isinstance(prompt, str) else " ".join(texts).strip()
            print("    prompt: " + text.replace("\n", " "))
        done.append(path.name)

    print()
    verb = "planned" if args.dry_run else "dressed"
    print(f"Summary     : {len(done)} {verb}, {len(failed)} failed")
    if failed:
        print("Failed      : " + ", ".join(failed))
    if not args.dry_run and done:
        print(f"Results     : {host_dest if host_dest else OUTPUT_DIR / prefix_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
