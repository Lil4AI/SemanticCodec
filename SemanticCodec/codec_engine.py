#!/usr/bin/env python3
"""SemanticCodec v1.0.0 (Final Research Prototype) — Core Engine.

This module is the single source of truth for everything the codec does.
``app.py`` (Flask Web UI) and ``cli.py`` (command line) are thin layers on top.

Architecture overview
---------------------
Encoder  (original image -> ``.semic.jpg``)
    1. Foreground extraction (rembg / U2-Net, GrabCut fallback)
    2. ROI coding: tight bounding-box crop of the foreground + alpha zeroing
    3. Morphological colour decontamination (edge bleed) of the foreground edge
    4. Background semantics: VQA-style BLIP captions + style classification
    5. Structure maps: Depth-Anything-V2 (depth) + SegFormer ADE20K (8-bit indexed PNG)
    6. Low-frequency colour/illumination context: tiny JPEG thumbnail
    7. Bitrate-aware profile (Lite / Standard / HighFi) controls every quantiser
    8. Polyglot container: JPEG preview + ZIP payload in one ``.semic.jpg`` file

Decoder  (``.semic.jpg`` -> image)
    Lite      : thumbnail background + crisp foreground (no AI, instant)
    Turbo     : LCM-LoRA, 4 steps (falls back to UniPC 8 steps when unavailable)
    Standard  : UniPC, 20 steps, Multi-ControlNet (Depth + Seg)
    HighFi    : UniPC, 50 steps, Multi-ControlNet (Depth + Seg)
    Post      : Reinhard LAB colour harmonisation, depth-layered bokeh,
                soft-alpha foreground compositing, EXIF provenance.

Heavy dependencies (torch / transformers / diffusers / rembg) are imported lazily,
so the container format, Lite decoding, ``inspect`` and the metric functions work
on any machine that only has numpy + Pillow + OpenCV installed.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import threading
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # optional: EXIF provenance tag on decoded images
    import piexif
except Exception:  # pragma: no cover - optional dependency
    piexif = None

try:  # optional: everything AI related
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


# =============================================================================
# Constants
# =============================================================================

VERSION = "1.0.0"
MODEL_VERSION = "sd15-inpaint_multi-cn-depth-seg_v1.0"
CONTAINER_FORMAT = "semic-polyglot (JPEG preview + ZIP payload)"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(BASE_DIR, "loras")

# --- Model identifiers -------------------------------------------------------
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SEG_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
CN_DEPTH_MODEL_ID = "lllyasviel/control_v11f1p_sd15_depth"
CN_SEG_MODEL_ID = "lllyasviel/control_v11p_sd15_seg"
# runwayml/* was removed from the Hub in 2024; keep it first (local caches still
# have it) and fall back to the official community mirror.
SD_INPAINT_MODEL_IDS = (
    "runwayml/stable-diffusion-inpainting",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
)
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

# --- Container member names --------------------------------------------------
MANIFEST_NAME = "manifest.json"
FG_COLOR_NAME = "fg_color.webp"
FG_ALPHA_NAME = "fg_alpha.webp"
THUMB_NAME = "thumb.jpg"
DEPTH_NAME = "depth_map.webp"
DEPTH_LEGACY_NAME = "depth_map.png"
SEG_NAME = "seg_map.png"
SEG_LEGACY_NAME = "seg_map.webp"

# --- Processing parameters ---------------------------------------------------
ALPHA_NOISE_FLOOR = 8          # alpha < 8 is treated as 0 (rembg speckles)
FG_MIN_AREA_RATIO = 0.001      # < 0.1 % non-zero alpha => "no foreground"
CORE_ALPHA_THRESHOLD = 210     # alpha >= 210 is certain foreground interior
EDGE_BLEED_PX = 4              # colour bleed distance outside the alpha edge
MAX_INFERENCE_PIXELS = 768 * 768   # 8 GB VRAM safe limit for Multi-ControlNet
BG_INPUT_BLUR_RADIUS = 15      # blur applied to the upscaled thumbnail before SD
PREVIEW_BG_COLOR = (20, 20, 20)

DEFAULT_PROMPT = (
    "a natural background environment, photorealistic, highly detailed, "
    "sharp focus, 8k"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, painting, abstract, artificial, low quality, deformed"


# ADE20K palette (150 classes) used by lllyasviel/control_v11p_sd15_seg.
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)
ADE20K_NUM_CLASSES = int(ADE20K_PALETTE.shape[0])  # 150
# 256 x 3 = 768 entries, zero padded, for PIL.Image.putpalette()
ADE20K_FLAT_PALETTE: List[int] = (
    ADE20K_PALETTE.flatten().tolist() + [0] * (768 - ADE20K_PALETTE.size)
)


# =============================================================================
# Profiles
# =============================================================================

ENCODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Lite": {
        "preview_max_dim": 256,
        "preview_quality": 60,
        "thumb_max_dim": 128,
        "thumb_quality": 6,
        "fg_color_quality": 72,
        "fg_alpha_lossless": False,
        "fg_alpha_quality": 80,
        "depth_max_dim": 256,
        "depth_lossless": False,
        "depth_quality": 55,
        "include_seg": False,
        "seg_max_dim": 0,
    },
    "Standard": {
        "preview_max_dim": 384,
        "preview_quality": 75,
        "thumb_max_dim": 256,
        "thumb_quality": 8,
        "fg_color_quality": 84,
        "fg_alpha_lossless": True,
        "fg_alpha_quality": 100,
        "depth_max_dim": 384,
        "depth_lossless": False,
        "depth_quality": 70,
        "include_seg": True,
        "seg_max_dim": 384,
    },
    "HighFi": {
        "preview_max_dim": 512,
        "preview_quality": 82,
        "thumb_max_dim": 256,
        "thumb_quality": 12,
        "fg_color_quality": 92,
        "fg_alpha_lossless": True,
        "fg_alpha_quality": 100,
        "depth_max_dim": 512,
        "depth_lossless": True,
        "depth_quality": 100,
        "include_seg": True,
        "seg_max_dim": 512,
    },
}

# ``steps`` = *effective* denoising steps (what actually runs after ``strength``).
DECODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Lite": {"steps": 0, "label": "0 steps - Instant Thumbnail + Crisp FG"},
    "Turbo": {
        "steps": 4, "guidance_scale": 1.8, "strength": 0.85, "use_lcm": True,
        "fallback_steps": 8, "fallback_guidance_scale": 7.5,
        "label": "4 steps - LCM-LoRA",
    },
    "Standard": {
        "steps": 20, "guidance_scale": 7.5, "strength": 0.85, "use_lcm": False,
        "label": "20 steps - UniPC Balanced",
    },
    "HighFi": {
        "steps": 50, "guidance_scale": 7.5, "strength": 0.85, "use_lcm": False,
        "label": "50 steps - Full Multi-ControlNet",
    },
}

DEFAULT_STRENGTH = 0.85
DEFAULT_COLOR_LOCK = 0.6

ProgressFn = Optional[Callable[..., None]]


def _notify(progress: ProgressFn, *args: Any) -> None:
    if progress is not None:
        try:
            progress(*args)
        except Exception:
            pass


# =============================================================================
# Generic helpers
# =============================================================================

def load_image(source: Union[str, bytes, Image.Image, io.IOBase]) -> Tuple[Image.Image, Optional[int]]:
    """Load an image as RGB (EXIF orientation applied). Returns (image, file_size)."""
    file_size: Optional[int] = None
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        file_size = len(source)
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, str):
        file_size = os.path.getsize(source)
        img = Image.open(source)
    else:
        data = source.read()
        file_size = len(data)
        img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[3])
        img = canvas
    return img.convert("RGB"), file_size


def fit_within(width: int, height: int, max_dim: int) -> Tuple[int, int]:
    """Size that fits inside ``max_dim`` keeping aspect ratio (never upscales)."""
    if max_dim <= 0 or max(width, height) <= max_dim:
        return width, height
    s = max_dim / float(max(width, height))
    return max(1, int(round(width * s))), max(1, int(round(height * s)))


def inference_size(width: int, height: int, max_pixels: int = MAX_INFERENCE_PIXELS) -> Tuple[int, int]:
    """Multiple-of-8 processing size, capped at ``max_pixels`` (VRAM safety)."""
    w, h = width, height
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / float(w * h))
        w, h = w * s, h * s
    return max(8, int(w) // 8 * 8), max(8, int(h) // 8 * 8)


def _encode_image(img: Image.Image, fmt: str, **kwargs: Any) -> bytes:
    """Encode PIL Image to bytes with format-specific mode sanitization.

    - WEBP: requires RGB, RGBA, or RGBX mode (converts L / P / CMYK automatically).
    - JPEG: requires RGB or L mode (converts RGBA / P / CMYK automatically).
    """
    buf = io.BytesIO()
    save_img = img
    fmt_upper = fmt.upper()
    if fmt_upper == "WEBP":
        if save_img.mode not in ("RGB", "RGBA", "RGBX"):
            save_img = save_img.convert("RGBA" if "A" in save_img.mode else "RGB")
    elif fmt_upper in ("JPEG", "JPG"):
        if save_img.mode not in ("RGB", "L"):
            save_img = save_img.convert("RGB")
    save_img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr + 0.5, 0, 255).astype(np.uint8)


def format_bytes(n: Optional[float]) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# =============================================================================
# Foreground: extraction, ROI crop, colour decontamination
# =============================================================================

def fallback_foreground_alpha(image: Image.Image) -> np.ndarray:
    """GrabCut based foreground mask, used when rembg is not installed."""
    w, h = image.size
    sw, sh = fit_within(w, h, 400)
    small = np.asarray(image.resize((sw, sh), Image.BILINEAR))[:, :, ::-1].copy()
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(sw * 0.12), int(sh * 0.08), max(1, int(sw * 0.76)), max(1, int(sh * 0.88)))
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(fg, (0, 0), 1.5)


def _edge_bleed(rgb: np.ndarray, known: np.ndarray, target: np.ndarray, max_iter: int = 512) -> np.ndarray:
    """Propagate colours from ``known`` pixels outward until ``target`` is covered.

    Normalised-convolution dilation: every iteration fills the next 1px ring with
    the average colour of its already-known neighbours. Only foreground-interior
    colours are used as sources, so the original background colour can never leak
    back into the semi-transparent edge.
    """
    acc = rgb.astype(np.float32) * known[..., None]
    wgt = known.astype(np.float32)
    known_cur = known.copy()
    remaining = target & ~known_cur
    it = 0
    while remaining.any() and it < max_iter:
        num = cv2.boxFilter(acc, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REPLICATE)
        den = cv2.boxFilter(wgt, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REPLICATE)
        newly = (den > 0.5) & ~known_cur
        if not newly.any():
            break
        acc[newly] = num[newly] / den[newly][:, None]
        wgt[newly] = 1.0
        known_cur |= newly
        remaining = target & ~known_cur
        it += 1
    if remaining.any():
        fill = rgb[known].mean(axis=0) if known.any() else np.zeros(3, np.float32)
        acc[remaining] = fill
    return _to_uint8(acc)


def decontaminate_edges(rgb_crop: np.ndarray, alpha_crop: np.ndarray,
                        core_threshold: int = CORE_ALPHA_THRESHOLD,
                        bleed_px: int = EDGE_BLEED_PX) -> np.ndarray:
    """Morphological colour decontamination of a foreground crop.

    * core  (alpha >= 210)            : original colours are kept
    * ring  (0 < alpha < 210) + 4px   : replaced by colours bled from the core
    * far   (alpha == 0, > 4px away)  : zeroed (entropy -> 0, WebP friendly)
    """
    core = alpha_crop >= core_threshold
    if not core.any():
        peak = int(alpha_crop.max())
        core = alpha_crop >= max(1, int(peak * 0.8))
    visible = (alpha_crop > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bleed_px + 1, 2 * bleed_px + 1))
    near = cv2.dilate(visible, k).astype(bool)
    fill = near & ~core
    out = np.zeros_like(rgb_crop)
    out[core] = rgb_crop[core]
    if fill.any():
        bled = _edge_bleed(rgb_crop, core, fill)
        out[fill] = bled[fill]
    return out


@dataclass
class ForegroundResult:
    has_foreground: bool
    bbox: Optional[List[int]]          # [x_min, y_min, x_max, y_max), max exclusive
    rgb_crop: Optional[Image.Image]    # decontaminated RGB crop
    alpha_crop: Optional[Image.Image]  # L crop
    alpha_full: np.ndarray             # (H, W) uint8 full-canvas alpha
    method: str


def compute_alpha_bbox(alpha: np.ndarray, min_area_ratio: float = FG_MIN_AREA_RATIO) -> Optional[List[int]]:
    nz = alpha > 0
    if nz.sum() < max(1, min_area_ratio * alpha.size):
        return None
    ys = np.flatnonzero(nz.any(axis=1))
    xs = np.flatnonzero(nz.any(axis=0))
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def extract_and_optimize_foreground(original_img: Image.Image,
                                    alpha: Optional[np.ndarray] = None,
                                    models: Optional["ModelManager"] = None) -> ForegroundResult:
    """ROI coding of the foreground: tight bbox crop + alpha zeroing + edge bleed."""
    rgb = np.asarray(original_img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    method = "external"
    if alpha is None:
        if models is not None:
            alpha, method = models.foreground_alpha(original_img)
        else:
            alpha, method = fallback_foreground_alpha(original_img), "grabcut"
    alpha = np.asarray(alpha, dtype=np.uint8)
    if alpha.ndim == 3:
        alpha = alpha[..., -1]
    if alpha.shape != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = alpha.copy()
    alpha[alpha < ALPHA_NOISE_FLOOR] = 0

    bbox = compute_alpha_bbox(alpha)
    if bbox is None:
        return ForegroundResult(False, None, None, None, np.zeros((h, w), np.uint8), method)

    x0, y0, x1, y1 = bbox
    rgb_crop = rgb[y0:y1, x0:x1]
    a_crop = alpha[y0:y1, x0:x1]
    clean = decontaminate_edges(rgb_crop, a_crop)
    return ForegroundResult(
        True, bbox, Image.fromarray(clean, "RGB"), Image.fromarray(a_crop, "L"), alpha, method
    )


# =============================================================================
# Segmentation map: 8-bit indexed palette PNG
# =============================================================================

def seg_ids_to_png_bytes(seg_ids: np.ndarray) -> bytes:
    """Store class IDs (0..149) as an 8-bit palette PNG (mode "P")."""
    ids = np.clip(np.asarray(seg_ids), 0, ADE20K_NUM_CLASSES - 1).astype(np.uint8)
    h, w = ids.shape
    im = Image.frombytes("P", (w, h), np.ascontiguousarray(ids).tobytes())
    im.putpalette(ADE20K_FLAT_PALETTE)
    return _encode_image(im, "PNG", optimize=True)


def seg_ids_to_image(seg_ids: np.ndarray) -> Image.Image:
    ids = np.clip(np.asarray(seg_ids), 0, ADE20K_NUM_CLASSES - 1).astype(np.uint8)
    h, w = ids.shape
    im = Image.frombytes("P", (w, h), np.ascontiguousarray(ids).tobytes())
    im.putpalette(ADE20K_FLAT_PALETTE)
    return im


def prepare_seg_control(seg_map: Optional[Image.Image], size: Tuple[int, int]) -> Optional[Image.Image]:
    """Resize with NEAREST *before* palette->RGB conversion (no fake boundary colours)."""
    if seg_map is None:
        return None
    if seg_map.mode == "P":
        return seg_map.resize(size, Image.NEAREST).convert("RGB")
    return seg_map.convert("RGB").resize(size, Image.NEAREST)  # legacy RGB WebP map


# =============================================================================
# Model manager (VRAM stage separation)
# =============================================================================

class ModelManager:
    """Loads AI models lazily and keeps encoder / decoder VRAM usage exclusive.

    8 GB VRAM strategy: the encoder models (BLIP, SegFormer) are moved to the CPU
    as soon as an encode finishes and the decoder (SD-Inpaint + 2 ControlNets) uses
    ``enable_model_cpu_offload``. All public inference entry points must be called
    while holding ``self.lock``.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.lock = threading.Lock()
        self.verbose = verbose
        # encoder side
        self.blip_processor = None
        self.blip_model = None
        self.depth_estimator = None
        self.seg_feature_extractor = None
        self.seg_model = None
        self.rembg_session = None
        self._rembg_failed = False
        self._encoder_attempted = False
        self._encoder_on_gpu = False
        # decoder side
        self.controlnet_depth = None
        self.controlnet_seg = None
        self.sd_pipeline = None
        self.unipc_scheduler = None
        self.lcm_scheduler = None
        self.lcm_lora_loaded = False
        self.lcm_error: Optional[str] = None
        self._decoder_attempted = False
        self.decoder_error: Optional[str] = None
        self.encoder_errors: Dict[str, str] = {}

    # -- environment --------------------------------------------------------
    @property
    def torch_available(self) -> bool:
        return torch is not None

    @property
    def cuda(self) -> bool:
        return torch is not None and torch.cuda.is_available()

    @property
    def device(self) -> str:
        return "cuda" if self.cuda else "cpu"

    @property
    def dtype(self):
        if torch is None:
            return None
        return torch.float16 if self.cuda else torch.float32

    @property
    def vram_gb(self) -> float:
        if not self.cuda:
            return 0.0
        try:
            return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        except Exception:
            return 0.0

    @property
    def high_vram(self) -> bool:
        """True on GPUs with >= 16 GB VRAM (e.g. A100 40GB/80GB, L4 24GB, RTX 3090/4090)."""
        if os.environ.get("SEMIC_HIGH_VRAM", "").strip() == "1":
            return self.cuda
        return self.vram_gb >= 16.0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SemanticCodec] {msg}", flush=True)

    def _empty_cache(self) -> None:
        if self.cuda:
            torch.cuda.empty_cache()

    def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "version": VERSION,
            "torch": getattr(torch, "__version__", None),
            "cuda": self.cuda,
            "device_name": None,
            "vram_total_gb": None,
            "encoder": {
                "blip": self.blip_model is not None,
                "depth": self.depth_estimator is not None,
                "segformer": self.seg_model is not None,
                "rembg": self.rembg_session is not None,
                "on_gpu": self._encoder_on_gpu,
                "errors": self.encoder_errors,
            },
            "decoder": {
                "loaded": self.sd_pipeline is not None,
                "lcm_lora": self.lcm_lora_loaded,
                "lcm_error": self.lcm_error,
                "error": self.decoder_error,
            },
        }
        if self.cuda:
            try:
                info["device_name"] = torch.cuda.get_device_name(0)
                info["vram_total_gb"] = self.vram_gb
                info["high_vram_mode"] = self.high_vram
            except Exception:
                pass
        return info

    # -- encoder ------------------------------------------------------------
    def load_encoder_models(self) -> bool:
        """Load (or move back to GPU) BLIP, Depth-Anything-V2 and SegFormer."""
        if torch is None:
            self.encoder_errors["torch"] = "PyTorch is not installed"
            return False
        if self.sd_pipeline is not None:
            self._empty_cache()

        if not self._encoder_attempted:
            self._encoder_attempted = True
            self._log("Loading encoder models (BLIP / Depth-Anything-V2 / SegFormer)...")
            try:
                from transformers import (BlipForConditionalGeneration, BlipProcessor,
                                          SegformerForSemanticSegmentation,
                                          SegformerImageProcessor)
                from transformers import pipeline as hf_pipeline
            except Exception as e:
                self.encoder_errors["transformers"] = str(e)
                self._log(f"transformers unavailable: {e}")
                return False

            try:
                self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    BLIP_MODEL_ID, torch_dtype=self.dtype).to(self.device)
                self.blip_model.eval()
            except Exception as e:
                self.blip_model = None
                self.encoder_errors["blip"] = str(e)
                self._log(f"BLIP failed to load: {e}")

            try:
                self.depth_estimator = hf_pipeline(
                    "depth-estimation", model=DEPTH_MODEL_ID, device=0 if self.cuda else -1)
            except Exception as e:
                self.depth_estimator = None
                self.encoder_errors["depth"] = str(e)
                self._log(f"Depth-Anything-V2 failed to load: {e}")

            try:
                self.seg_feature_extractor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
                self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
                    SEG_MODEL_ID, torch_dtype=self.dtype).to(self.device)
                self.seg_model.eval()
            except Exception as e:
                self.seg_model = None
                self.encoder_errors["segformer"] = str(e)
                self._log(f"SegFormer failed to load: {e}")

            if self.cuda:
                if self.high_vram:
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    except Exception:
                        pass
                self._log(f"CUDA device: {torch.cuda.get_device_name(0)} ({self.vram_gb} GB VRAM, high_vram={self.high_vram})")
        elif not self._encoder_on_gpu and self.cuda:
            for m in (self.blip_model, self.seg_model):
                if m is not None:
                    m.to(self.device)

        self._encoder_on_gpu = self.cuda
        return any(m is not None for m in (self.blip_model, self.depth_estimator, self.seg_model))

    def offload_encoder_to_cpu(self) -> None:
        """Free ~1.2 GB VRAM for the decoder by parking BLIP + SegFormer on the CPU (skipped on >=16GB High-VRAM GPUs)."""
        if not self.cuda or not self._encoder_on_gpu or self.high_vram:
            return
        for m in (self.blip_model, self.seg_model):
            if m is not None:
                m.to("cpu")
        self._encoder_on_gpu = False
        self._empty_cache()
        self._log("Encoder models offloaded to CPU.")

    def foreground_alpha(self, image: Image.Image) -> Tuple[np.ndarray, str]:
        if not self._rembg_failed:
            try:
                from rembg import new_session, remove
                if self.rembg_session is None:
                    try:
                        self.rembg_session = new_session("u2net")
                    except Exception:
                        self.rembg_session = None
                rgba = remove(image, session=self.rembg_session) if self.rembg_session else remove(image)
                if not isinstance(rgba, Image.Image):
                    rgba = Image.open(io.BytesIO(rgba))
                return np.asarray(rgba.convert("RGBA"))[..., 3].copy(), "rembg-u2net"
            except Exception as e:
                self._rembg_failed = True
                self.encoder_errors["rembg"] = str(e)
                self._log(f"rembg unavailable ({e}); falling back to GrabCut.")
        return fallback_foreground_alpha(image), "grabcut"

    def generate_depth_map(self, image: Image.Image) -> Optional[Image.Image]:
        """Depth-Anything-V2 relative depth (255 = near, 0 = far) as L image."""
        if self.depth_estimator is None:
            return None
        safe_img = image.convert("RGB")
        result = self.depth_estimator(safe_img)
        depth = result["depth"]
        if depth.size != safe_img.size:
            depth = depth.resize(safe_img.size, Image.BICUBIC)
        return depth.convert("L")

    def generate_seg_ids(self, image: Image.Image, max_dim: int) -> Optional[np.ndarray]:
        """SegFormer ADE20K class IDs at <= ``max_dim`` (aspect ratio kept)."""
        if self.seg_model is None or self.seg_feature_extractor is None:
            return None
        safe_img = image.convert("RGB")
        tw, th = fit_within(safe_img.width, safe_img.height, max_dim)
        inputs = self.seg_feature_extractor(images=safe_img, return_tensors="pt")
        inputs = {k: v.to(self.seg_model.device, self.seg_model.dtype) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.seg_model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(th, tw), mode="bilinear", align_corners=False)
            ids = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return ids

    def generate_seg_map(self, image: Image.Image, max_dim: int = 512) -> Optional[Image.Image]:
        """Generate an 8-bit indexed ADE20K palette ``Image`` (mode ``"P"``)."""
        ids = self.generate_seg_ids(image, max_dim)
        return seg_ids_to_image(ids) if ids is not None else None

    def generate_vqa_caption(self, image: Image.Image, bg_mask_img: Image.Image) -> Tuple[str, str, str, str]:
        """VQA-style multi-angle background prompt + style classification.

        Returns (positive_prompt, negative_prompt, style, source).
        """
        if self.blip_model is None or self.blip_processor is None:
            return (clean_and_truncate_prompt(DEFAULT_PROMPT), DEFAULT_NEGATIVE_PROMPT,
                    "photo", "fallback")

        proc, model = self.blip_processor, self.blip_model

        def ask(img: Image.Image, text: str, max_new_tokens: int) -> str:
            inputs = proc(img, text=text, return_tensors="pt").to(model.device, model.dtype)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return proc.decode(out[0], skip_special_tokens=True).strip()

        safe_img = image.convert("RGB")
        style_prefix = "The artistic style of this image is best described as a"
        raw_style = ask(safe_img, style_prefix, 20).lower()
        detected = categorize_style(raw_style)
        if not detected:
            qa_prompt = "Question: Is this image a photograph, a digital illustration, or a painting? Answer:"
            qa_answer = ask(safe_img, qa_prompt, 10).lower().replace(qa_prompt.lower(), "").strip()
            detected = categorize_style(qa_answer) or "photo"
            self._log(f"[VQA Style] stage-1 inconclusive, stage-2 '{qa_answer}' -> {detected}")
        else:
            self._log(f"[VQA Style] '{raw_style}' -> {detected}")

        # Foreground blacked out so BLIP describes the background, not the subject.
        bg_only = Image.composite(safe_img, Image.new("RGB", safe_img.size, (0, 0, 0)), bg_mask_img.convert("L"))
        prefixes = [
            "the background scene shows",
            "the weather and lighting is",
            "the ground or surface is",
            "in the distance there is",
        ]
        captions = [ask(bg_only, p, 30) for p in prefixes]
        pos, neg = build_style_prompts(", ".join(captions), detected)
        self._log(f"[VQA Prompt] {pos}")
        return pos, neg, detected, "blip-vqa"

    # -- decoder ------------------------------------------------------------
    def load_decoder_models(self) -> bool:
        """Load SD-Inpaint + Multi-ControlNet (Depth + Seg). Offloads the encoder first."""
        if torch is None:
            self.decoder_error = "PyTorch is not installed"
            return False
        self.offload_encoder_to_cpu()
        if self.sd_pipeline is not None:
            return True
        if self._decoder_attempted and self.decoder_error:
            return False
        self._decoder_attempted = True
        self._log("Loading decoder models (SD-Inpaint + ControlNet Depth/Seg)...")
        try:
            from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                                   UniPCMultistepScheduler)
            try:
                from diffusers import LCMScheduler
            except Exception:
                LCMScheduler = None

            kw = {"torch_dtype": torch.float16} if self.cuda else {}
            self.controlnet_depth = ControlNetModel.from_pretrained(CN_DEPTH_MODEL_ID, **kw)
            self.controlnet_seg = ControlNetModel.from_pretrained(CN_SEG_MODEL_ID, **kw)

            pipe, last_err = None, None
            for repo in SD_INPAINT_MODEL_IDS:
                try:
                    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        repo, controlnet=[self.controlnet_depth, self.controlnet_seg],
                        safety_checker=None, requires_safety_checker=False, **kw)
                    self._log(f"SD-Inpaint loaded from {repo}")
                    break
                except Exception as e:
                    last_err = e
            if pipe is None:
                raise last_err if last_err else RuntimeError("SD-Inpaint could not be loaded")

            base_cfg = pipe.scheduler.config
            self.unipc_scheduler = UniPCMultistepScheduler.from_config(base_cfg)
            self.lcm_scheduler = LCMScheduler.from_config(base_cfg) if LCMScheduler else None
            pipe.scheduler = self.unipc_scheduler

            if self.cuda:
                if self.high_vram:
                    try:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    except Exception:
                        pass
                    pipe.to("cuda")
                    self._log(f"High-VRAM mode ({self.vram_gb} GB): Full pipeline resident on CUDA + SDPA/TF32 enabled.")
                else:
                    pipe.enable_model_cpu_offload()
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        self._log("xformers memory-efficient attention enabled.")
                    except Exception:
                        pipe.enable_attention_slicing()
                        self._log("Attention slicing enabled (8GB VRAM safe mode).")
                    try:
                        pipe.vae.enable_slicing()
                    except Exception:
                        try:
                            pipe.enable_vae_slicing()
                        except Exception:
                            pass
            else:
                pipe.to("cpu")
                self._log("Running the decoder on CPU - this will be slow.")
            self.sd_pipeline = pipe
            self.decoder_error = None
            return True
        except Exception as e:
            self.decoder_error = str(e)
            self._log(f"Decoder failed to load: {e}")
            return False

    def _ensure_lcm_lora(self) -> bool:
        if self.lcm_lora_loaded:
            return True
        if self.lcm_error is not None or self.lcm_scheduler is None:
            if self.lcm_scheduler is None and self.lcm_error is None:
                self.lcm_error = "LCMScheduler not available in this diffusers version"
            return False
        try:
            self.sd_pipeline.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
            self.lcm_lora_loaded = True
            self._log("LCM-LoRA loaded (Turbo mode available).")
            return True
        except Exception as e:
            self.lcm_error = str(e)
            self._log(f"LCM-LoRA unavailable ({e}); Turbo falls back to UniPC 8 steps.")
            return False

    def configure_generation(self, use_lcm: bool, user_lora: str = "",
                             user_lora_scale: float = 1.0) -> Dict[str, Any]:
        """Select scheduler + LoRA adapters for one generation call."""
        pipe = self.sd_pipeline
        warnings: List[str] = []
        lcm_active = bool(use_lcm) and self._ensure_lcm_lora()
        if use_lcm and not lcm_active:
            warnings.append("LCM-LoRA unavailable: Turbo fell back to UniPC 8 steps")
        pipe.scheduler = self.lcm_scheduler if lcm_active else self.unipc_scheduler

        adapters: List[str] = []
        weights: List[float] = []
        if lcm_active:
            adapters.append("lcm")
            weights.append(1.0)
        user_active = False
        if user_lora:
            safe = os.path.basename(user_lora)
            if os.path.isfile(os.path.join(LORA_DIR, safe)):
                try:
                    pipe.load_lora_weights(LORA_DIR, weight_name=safe, adapter_name="user")
                    adapters.append("user")
                    weights.append(float(user_lora_scale))
                    user_active = True
                except Exception as e:
                    warnings.append(f"LoRA '{safe}' failed to load: {e}")
            else:
                warnings.append(f"LoRA '{safe}' not found in loras/")

        if adapters:
            try:
                pipe.enable_lora()
            except Exception:
                pass
            try:
                pipe.set_adapters(adapters, adapter_weights=weights)
            except Exception as e:
                warnings.append(f"set_adapters failed: {e}")
        elif self.lcm_lora_loaded:
            try:
                pipe.disable_lora()
            except Exception:
                pass
        return {"lcm": lcm_active, "user_lora": user_active, "warnings": warnings}

    def release_generation(self, cfg: Dict[str, Any]) -> None:
        if cfg.get("user_lora"):
            try:
                self.sd_pipeline.delete_adapters("user")
            except Exception:
                try:
                    self.sd_pipeline.unload_lora_weights()
                    self.lcm_lora_loaded = False
                except Exception:
                    pass
        self._empty_cache()


def generate_seg_map(image: Image.Image, max_dim: int = 512,
                     models: Optional[ModelManager] = None) -> Image.Image:
    """Return an 8-bit indexed ADE20K segmentation map (``mode="P"``) fitted within ``max_dim``.

    Uses SegFormer when available via ``models``, otherwise produces a spatial quantization map.
    """
    ids = models.generate_seg_ids(image, max_dim) if models is not None else None
    if ids is None:
        tw, th = fit_within(image.width, image.height, max_dim)
        small = np.asarray(image.resize((tw, th), Image.BILINEAR), dtype=np.uint16)
        ids = ((small[:, :, 0] // 52) * 25 + (small[:, :, 1] // 52) * 5 + (small[:, :, 2] // 52)).astype(np.uint8)
        ids = np.minimum(ids, 149)
    return seg_ids_to_image(ids)


# =============================================================================
# Prompt helpers
# =============================================================================

def categorize_style(text: str) -> Optional[str]:
    if any(w in text for w in ["photo", "photograph", "real", "camera", "picture", "captured", "dslr", "natural"]):
        return "photo"
    if any(w in text for w in ["illust", "anime", "cartoon", "drawing", "sketch", "2d", "comic", "manga",
                               "digital art", "artwork", "vector"]):
        return "illustration"
    if any(w in text for w in ["paint", "watercolor", "oil", "acrylic", "canvas", "brush", "stroke"]):
        return "painting"
    if any(w in text for w in ["pattern", "texture", "flat", "abstract", "geometric", "design"]):
        return "pattern"
    return None


def build_style_prompts(combined: str, style: str) -> Tuple[str, str]:
    if style == "photo":
        pos = combined + ", true photographic masterpiece, f/8, deep depth of field, sharp focus everywhere, photorealistic, 8k, highly detailed photography"
        neg = "blurry, shallow depth of field, bokeh, out of focus, painting, illustration, cartoon, fake, 3d render, abstract, oil painting, ugly, deformed, grain, noise"
    elif style == "illustration":
        pos = combined + ", highly detailed illustration, 2d anime style, sharp lines, vibrant colors, flat shading, masterpiece"
        neg = "photograph, realistic, 3d render, blurry, bokeh, painting, messy, distorted, photographic"
    elif style == "painting":
        pos = combined + ", beautiful painting, traditional art style, visible brushstrokes, canvas texture, masterpiece"
        neg = "photograph, realistic, 3d render, blurry, anime, cartoon, flat, ugly, photo"
    elif style == "pattern":
        pos = combined + ", seamless pattern, flat texture, 2d design, repeating elements, high quality graphic design"
        neg = "depth, 3d, realistic, blurry, bokeh, photograph, perspective, messy"
    else:
        pos = combined + ", masterpiece, high quality"
        neg = "blurry, low quality"
    return clean_and_truncate_prompt(pos), neg


def clean_and_truncate_prompt(prompt: str, max_tokens: int = 70) -> str:
    """Remove empty comma chunks and keep the prompt under the CLIP 77-token limit."""
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    kept: List[str] = []
    count = 0.0
    for p in parts:
        est = len(p.split()) * 1.5 + 1
        if count + est > max_tokens:
            break
        kept.append(p)
        count += est
    return ", ".join(kept)


# =============================================================================
# Metrics
# =============================================================================

def compute_bpp(size_bytes: int, width: int, height: int) -> float:
    return (size_bytes * 8.0) / float(max(1, width * height))


def compute_compression_ratio_pct(semic_size: int, original_size: Optional[int]) -> Optional[float]:
    if not original_size:
        return None
    return (1.0 - semic_size / float(original_size)) * 100.0


def compute_fg_area_ratio_pct(alpha_full: Optional[np.ndarray]) -> float:
    if alpha_full is None or alpha_full.size == 0:
        return 0.0
    return float((alpha_full > 128).mean() * 100.0)


def compute_metrics(semic_size: int, original_size: Optional[int], width: int, height: int,
                    alpha_full: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Rate metrics of an encoded file (bpp, compression ratio, foreground ratio)."""
    ratio = compute_compression_ratio_pct(semic_size, original_size)
    return {
        "bpp": round(compute_bpp(semic_size, width, height), 4),
        "compression_ratio_pct": None if ratio is None else round(ratio, 2),
        "ratio_display": (f"{original_size / semic_size:.1f}:1" if original_size and semic_size else "N/A"),
        "fg_area_ratio_pct": round(compute_fg_area_ratio_pct(alpha_full), 2),
    }


def compute_psnr(a: Union[Image.Image, np.ndarray], b: Union[Image.Image, np.ndarray],
                 mask: Optional[np.ndarray] = None) -> Optional[float]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    diff = (x - y) ** 2
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return None
        diff = diff[m]
    mse = float(diff.mean())
    if mse <= 1e-10:
        return 99.0
    return round(10.0 * math.log10(255.0 ** 2 / mse), 3)


def compute_ssim(a: Union[Image.Image, np.ndarray], b: Union[Image.Image, np.ndarray]) -> float:
    """Gaussian-window SSIM (Wang et al. 2004) on luma."""
    def luma(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        return arr
    x, y = luma(a), luma(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return round(float(ssim_map.mean()), 4)


def histogram_correlation(a: Image.Image, b: Image.Image, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Masked 8x8x8 RGB histogram correlation in percent (0..100)."""
    x = np.asarray(a.convert("RGB"))
    y = np.asarray(b.convert("RGB").resize(a.size, Image.BILINEAR))
    m = None
    if mask is not None:
        m = cv2.resize(np.asarray(mask, np.uint8), a.size, interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        if m.sum() < 16:
            m = None
    hx = cv2.calcHist([x], [0, 1, 2], m, [8, 8, 8], [0, 256] * 3)
    hy = cv2.calcHist([y], [0, 1, 2], m, [8, 8, 8], [0, 256] * 3)
    corr = cv2.compareHist(hx, hy, cv2.HISTCMP_CORREL)
    if not np.isfinite(corr):
        return None
    return round(float(np.clip(corr, 0.0, 1.0) * 100.0), 2)


def compute_reconstruction_metrics(result: Image.Image, reference: Image.Image,
                                   alpha_full: Optional[np.ndarray]) -> Dict[str, Any]:
    if reference.size != result.size:
        reference = reference.resize(result.size, Image.LANCZOS)
    fg_mask = None if alpha_full is None else alpha_full > 128
    bg_mask = None if alpha_full is None else alpha_full <= 128
    return {
        "psnr_db": compute_psnr(result, reference),
        "ssim": compute_ssim(result, reference),
        "fg_psnr_db": None if fg_mask is None else compute_psnr(result, reference, fg_mask),
        "bg_psnr_db": None if bg_mask is None else compute_psnr(result, reference, bg_mask),
    }


# =============================================================================
# Encoder
# =============================================================================

@dataclass
class EncodeResult:
    data: bytes
    manifest: Dict[str, Any]
    stats: Dict[str, Any]
    lite_preview: Optional[Image.Image] = None


def _gradient_depth(width: int, height: int) -> Image.Image:
    """Fallback depth prior: far at the top, near at the bottom."""
    col = np.linspace(40, 215, height, dtype=np.float32)[:, None]
    return Image.fromarray(_to_uint8(np.repeat(col, width, axis=1)), "L")


def _package(preview_jpeg: bytes, members: List[Tuple[str, bytes]]) -> bytes:
    """Polyglot container: valid JPEG (preview) followed by a valid ZIP archive."""
    buf = io.BytesIO()
    buf.write(preview_jpeg)
    with zipfile.ZipFile(buf, "a") as zf:
        for name, payload in members:
            # images are already entropy coded -> store; JSON is deflated
            ctype = zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED
            zf.writestr(name, payload, compress_type=ctype)
    return buf.getvalue()


def encode_to_semic(source: Union[str, bytes, Image.Image], mode: str = "Standard", *,
                    models: Optional[ModelManager] = None,
                    use_ai: bool = True,
                    lora_model: str = "",
                    lora_scale: float = 1.0,
                    original_file_size: Optional[int] = None,
                    fg_alpha: Optional[np.ndarray] = None,
                    seg_ids: Optional[np.ndarray] = None,
                    depth_map: Optional[Image.Image] = None,
                    prompt: Optional[str] = None,
                    negative_prompt: Optional[str] = None,
                    image_style: Optional[str] = None,
                    seed: Optional[int] = None,
                    progress: ProgressFn = None) -> EncodeResult:
    """Encode an image into the ``.semic.jpg`` polyglot format.

    ``fg_alpha`` / ``seg_ids`` / ``depth_map`` let callers inject externally computed
    analysis results (user masks, offline tests); otherwise the AI models are used
    when ``use_ai`` is true, with deterministic fallbacks when they are unavailable.
    """
    if mode not in ENCODE_PROFILES:
        raise ValueError(f"Unknown encode mode '{mode}'. Choose from {list(ENCODE_PROFILES)}")
    prof = ENCODE_PROFILES[mode]
    t0 = time.time()

    original_img, detected_size = load_image(source)
    if original_file_size is None:
        original_file_size = detected_size
    width, height = original_img.size

    analysis: Dict[str, str] = {}
    ai = models if (use_ai and models is not None) else None
    lock = ai.lock if ai is not None else threading.Lock()

    with lock:
        ai_ready = ai.load_encoder_models() if ai is not None else False

        # 1. Foreground ROI
        _notify(progress, "foreground", 1, 6)
        fg = extract_and_optimize_foreground(original_img, alpha=fg_alpha, models=ai)
        analysis["foreground"] = fg.method
        bg_mask_img = Image.fromarray(255 - fg.alpha_full, "L")

        # 2. Background semantics
        _notify(progress, "caption", 2, 6)
        if prompt is not None:
            pos_prompt = clean_and_truncate_prompt(prompt)
            neg_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
            img_style = image_style or "photo"
            cap_src = "external"
        elif ai_ready:
            pos_prompt, neg_prompt, img_style, cap_src = ai.generate_vqa_caption(original_img, bg_mask_img)
        else:
            pos_prompt, neg_prompt, img_style, cap_src = (
                clean_and_truncate_prompt(DEFAULT_PROMPT), DEFAULT_NEGATIVE_PROMPT, "photo", "fallback")
        analysis["caption"] = cap_src

        # 3. Depth
        _notify(progress, "depth", 3, 6)
        if depth_map is not None:
            depth = depth_map.convert("L")
            analysis["depth"] = "external"
        else:
            depth = ai.generate_depth_map(original_img) if ai_ready else None
            analysis["depth"] = "depth-anything-v2-small" if depth is not None else "fallback-gradient"
            if depth is None:
                depth = _gradient_depth(width, height)
        depth = depth.resize(fit_within(width, height, prof["depth_max_dim"]), Image.BICUBIC) \
            if max(depth.size) > prof["depth_max_dim"] else depth

        # 4. Segmentation (8-bit indexed palette PNG)
        _notify(progress, "segmentation", 4, 6)
        seg_png: Optional[bytes] = None
        if prof["include_seg"]:
            ids = None
            if seg_ids is not None:
                ids = np.asarray(seg_ids, dtype=np.uint8)
                analysis["segmentation"] = "external"
            elif ai_ready:
                ids = ai.generate_seg_ids(original_img, prof["seg_max_dim"])
                analysis["segmentation"] = "segformer-b0-ade20k" if ids is not None else "unavailable"
            else:
                analysis["segmentation"] = "unavailable"
            if ids is not None:
                tw, th = fit_within(width, height, prof["seg_max_dim"])
                if ids.shape != (th, tw):
                    ids = cv2.resize(ids, (tw, th), interpolation=cv2.INTER_NEAREST)
                seg_png = seg_ids_to_png_bytes(ids)
        else:
            analysis["segmentation"] = "skipped (profile)"

        if ai is not None:
            ai.offload_encoder_to_cpu()

    # 5. Quantise / serialise components
    _notify(progress, "packaging", 5, 6)
    members: List[Tuple[str, bytes]] = []
    components: Dict[str, int] = {}
    if fg.has_foreground:
        fg_color_bytes = _encode_image(fg.rgb_crop, "WEBP", quality=prof["fg_color_quality"], method=6)
        if prof["fg_alpha_lossless"]:
            fg_alpha_bytes = _encode_image(fg.alpha_crop, "WEBP", lossless=True, quality=100, method=6)
        else:
            fg_alpha_bytes = _encode_image(fg.alpha_crop, "WEBP", quality=prof["fg_alpha_quality"], method=6)
        members += [(FG_COLOR_NAME, fg_color_bytes), (FG_ALPHA_NAME, fg_alpha_bytes)]
        components["fg_color_webp"] = len(fg_color_bytes)
        components["fg_alpha_webp"] = len(fg_alpha_bytes)

    thumb = original_img.copy()
    thumb.thumbnail((prof["thumb_max_dim"], prof["thumb_max_dim"]), Image.LANCZOS)
    thumb_bytes = _encode_image(thumb, "JPEG", quality=prof["thumb_quality"], optimize=True)
    members.append((THUMB_NAME, thumb_bytes))
    components["thumb_jpg"] = len(thumb_bytes)

    if prof["depth_lossless"]:
        depth_bytes = _encode_image(depth, "WEBP", lossless=True, quality=100, method=6)
    else:
        depth_bytes = _encode_image(depth, "WEBP", quality=prof["depth_quality"], method=6)
    members.append((DEPTH_NAME, depth_bytes))
    components["depth_map_webp"] = len(depth_bytes)

    if seg_png is not None:
        members.append((SEG_NAME, seg_png))
        components["seg_map_png"] = len(seg_png)

    # Polyglot preview: crisp foreground over a flat dark canvas (compresses to ~nothing)
    pw, ph = fit_within(width, height, prof["preview_max_dim"])
    preview = Image.new("RGB", (width, height), PREVIEW_BG_COLOR)
    if fg.has_foreground:
        x0, y0, _, _ = fg.bbox
        preview.paste(fg.rgb_crop, (x0, y0), fg.alpha_crop)
    preview = preview.resize((pw, ph), Image.LANCZOS)
    preview_bytes = _encode_image(preview, "JPEG", quality=prof["preview_quality"], optimize=True)
    components["preview_jpg"] = len(preview_bytes)

    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    manifest: Dict[str, Any] = {
        "version": VERSION,
        "guide_type": "depth_seg" if seg_png is not None else "depth",
        "width": width,
        "height": height,
        "has_foreground": fg.has_foreground,
        "fg_bbox": fg.bbox,
        "prompt": pos_prompt,
        "negative_prompt": neg_prompt,
        "image_style": img_style,
        "random_seed": int(seed),
        "model_version": MODEL_VERSION,
        "encoded_mode": mode,
        "bpp": 0.0,
        "lora_model": lora_model or "",
        "lora_scale": float(lora_scale),
        "fg_alpha_lossless": bool(prof["fg_alpha_lossless"]),
        "analysis": analysis,
    }

    # Two-pass packaging so that the manifest carries the (near-exact) final bpp.
    data = _package(preview_bytes, [(MANIFEST_NAME, json.dumps(manifest).encode("utf-8"))] + members)
    manifest["bpp"] = round(compute_bpp(len(data), width, height), 4)
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    data = _package(preview_bytes, [(MANIFEST_NAME, manifest_bytes)] + members)
    components["manifest_json"] = len(manifest_bytes)

    semic_size = len(data)
    metrics = compute_metrics(semic_size, original_file_size, width, height, fg.alpha_full)
    stats = {
        "original_size": original_file_size,
        "semic_size": semic_size,
        "compression_ratio": metrics["compression_ratio_pct"],
        "compression_ratio_pct": metrics["compression_ratio_pct"],
        "ratio_display": metrics["ratio_display"],
        "bpp": metrics["bpp"],
        "fg_area_ratio_pct": metrics["fg_area_ratio_pct"],
        "width": width,
        "height": height,
        "mode": mode,
        "has_foreground": fg.has_foreground,
        "fg_bbox": fg.bbox,
        "components": components,
        "analysis": analysis,
        "encode_time_sec": round(time.time() - t0, 2),
    }
    _notify(progress, "done", 6, 6)
    return EncodeResult(data=data, manifest=manifest, stats=stats)


# =============================================================================
# Container reading
# =============================================================================

@dataclass
class SemicPayload:
    manifest: Dict[str, Any]
    thumb: Image.Image
    fg_color: Optional[Image.Image] = None
    fg_alpha: Optional[Image.Image] = None
    bbox: Optional[List[int]] = None
    depth_map: Optional[Image.Image] = None
    seg_map: Optional[Image.Image] = None
    seg_format: str = "none"
    components: Dict[str, int] = field(default_factory=dict)
    total_size: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return int(self.manifest["width"])

    @property
    def height(self) -> int:
        return int(self.manifest["height"])

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "0.1.0"))

    @property
    def has_foreground(self) -> bool:
        return self.fg_color is not None and self.fg_alpha is not None and self.bbox is not None


def load_semic(source: Union[str, bytes, io.IOBase]) -> SemicPayload:
    """Read any ``.semic`` / ``.semic.jpg`` (v0.1.0 - v1.0.0)."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, str):
        with open(source, "rb") as f:
            data = f.read()
    else:
        data = source.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile as e:
        raise ValueError("Not a valid .semic container (ZIP payload not found)") from e

    with zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise ValueError("manifest.json missing from .semic container")
        manifest = json.loads(zf.read(MANIFEST_NAME))
        infos = zf.infolist()
        components = {i.filename: i.compress_size for i in infos}
        if infos:
            components["(preview jpeg)"] = min(i.header_offset for i in infos)

        def read_img(name: str) -> Optional[Image.Image]:
            if name not in names:
                return None
            im = Image.open(io.BytesIO(zf.read(name)))
            im.load()
            return im

        thumb = read_img(THUMB_NAME)
        if thumb is None:
            raise ValueError("thumb.jpg missing from .semic container")
        payload = SemicPayload(manifest=manifest, thumb=thumb.convert("RGB"),
                               components=components, total_size=len(data))

        w, h = payload.width, payload.height
        fg_color = read_img(FG_COLOR_NAME)
        fg_alpha = read_img(FG_ALPHA_NAME)
        if manifest.get("has_foreground", True) and fg_color is not None and fg_alpha is not None:
            bbox = manifest.get("fg_bbox") or [0, 0, w, h]   # legacy: full canvas
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            bw, bh = x1 - x0, y1 - y0
            fg_color = fg_color.convert("RGB")
            fg_alpha = fg_alpha.convert("L")
            if fg_color.size != (bw, bh):
                payload.warnings.append(f"fg_color size {fg_color.size} != bbox {(bw, bh)}; resized")
                fg_color = fg_color.resize((bw, bh), Image.LANCZOS)
            if fg_alpha.size != (bw, bh):
                fg_alpha = fg_alpha.resize((bw, bh), Image.LANCZOS)
            if not manifest.get("fg_alpha_lossless", True):
                a = np.asarray(fg_alpha).copy()
                a[a < 4] = 0
                a[a > 251] = 255
                fg_alpha = Image.fromarray(a, "L")
            payload.fg_color, payload.fg_alpha, payload.bbox = fg_color, fg_alpha, [x0, y0, x1, y1]

        depth = read_img(DEPTH_NAME) or read_img(DEPTH_LEGACY_NAME)
        payload.depth_map = depth.convert("L") if depth is not None else None

        seg = read_img(SEG_NAME)
        if seg is not None:
            payload.seg_map = seg if seg.mode == "P" else seg.convert("RGB")
            payload.seg_format = "png-indexed" if seg.mode == "P" else "png-rgb"
        else:
            seg = read_img(SEG_LEGACY_NAME)
            if seg is not None:
                payload.seg_map = seg.convert("RGB")
                payload.seg_format = "webp-rgb (legacy)"
    return payload


def build_alpha_canvas(payload: SemicPayload) -> np.ndarray:
    alpha = np.zeros((payload.height, payload.width), np.uint8)
    if payload.has_foreground:
        x0, y0, x1, y1 = payload.bbox
        alpha[y0:y1, x0:x1] = np.asarray(payload.fg_alpha)
    return alpha


def composite_foreground(background: Image.Image, payload: SemicPayload) -> Image.Image:
    """Soft-alpha compositing of the stored foreground at its exact bbox position."""
    bg = background.convert("RGB")
    if bg.size != (payload.width, payload.height):
        bg = bg.resize((payload.width, payload.height), Image.LANCZOS)
    if not payload.has_foreground:
        return bg
    out = np.asarray(bg, dtype=np.float32).copy()
    x0, y0, x1, y1 = payload.bbox
    a = np.asarray(payload.fg_alpha, dtype=np.float32)[..., None] / 255.0
    fgc = np.asarray(payload.fg_color, dtype=np.float32)
    out[y0:y1, x0:x1] = fgc * a + out[y0:y1, x0:x1] * (1.0 - a)
    return Image.fromarray(_to_uint8(out), "RGB")


def build_base_background(payload: SemicPayload, blur_radius: Optional[float] = None) -> Image.Image:
    """Thumbnail upscaled to full size; optional blur hides JPEG block artefacts."""
    base = payload.thumb.resize((payload.width, payload.height), Image.LANCZOS)
    if blur_radius is None:
        blur_radius = min(8.0, 0.5 * payload.width / float(max(1, payload.thumb.width)))
    if blur_radius > 0:
        base = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return base


def decode_lite(payload: SemicPayload) -> Image.Image:
    return composite_foreground(build_base_background(payload), payload)


def lite_preview_data_url(payload: SemicPayload, max_dim: int = 1280, quality: int = 85) -> str:
    img = decode_lite(payload)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    b64 = base64.b64encode(_encode_image(img, "JPEG", quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def inspect_semic(source: Union[str, bytes, io.IOBase, SemicPayload]) -> Dict[str, Any]:
    payload = source if isinstance(source, SemicPayload) else load_semic(source)
    m = payload.manifest
    return {
        "version": payload.version,
        "width": payload.width,
        "height": payload.height,
        "encoded_mode": m.get("encoded_mode", "Standard"),
        "guide_type": m.get("guide_type", "depth"),
        "has_foreground": payload.has_foreground,
        "fg_bbox": payload.bbox,
        "prompt": m.get("prompt", ""),
        "negative_prompt": m.get("negative_prompt", ""),
        "image_style": m.get("image_style", "photo"),
        "random_seed": m.get("random_seed"),
        "model_version": m.get("model_version", ""),
        "lora_model": m.get("lora_model", ""),
        "lora_scale": m.get("lora_scale", 1.0),
        "seg_format": payload.seg_format,
        "total_size": payload.total_size,
        "bpp": round(compute_bpp(payload.total_size, payload.width, payload.height), 4),
        "fg_area_ratio_pct": round(compute_fg_area_ratio_pct(build_alpha_canvas(payload)), 2),
        "components": payload.components,
        "warnings": payload.warnings,
        "manifest": m,
    }


# =============================================================================
# Decoder post-processing
# =============================================================================

def _lab(img01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img01.astype(np.float32), cv2.COLOR_RGB2LAB)


def harmonize_color_lab(generated: Image.Image, reference: Image.Image,
                        mask: Optional[np.ndarray] = None, strength: float = DEFAULT_COLOR_LOCK,
                        lowpass: Optional[Callable[[Image.Image], Image.Image]] = None,
                        eps: float = 1e-4) -> Image.Image:
    """Reinhard colour transfer in CIE L*a*b*, blended by ``strength``.

    ``C = (1-a) C_gen + a ( sigma_ref / (sigma_gen + eps) (C_gen - mu_gen) + mu_ref )``

    The reference (upscaled + blurred thumbnail) only carries low frequencies, so the
    generator statistics are measured on an equally low-passed copy (``lowpass``).
    Otherwise sigma_ref / sigma_gen would collapse and flatten the generated texture.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return generated
    gen_img = generated.convert("RGB")
    ref_img = reference.convert("RGB").resize(gen_img.size, Image.BILINEAR)
    gen = np.asarray(gen_img, np.float32) / 255.0
    ref = np.asarray(ref_img, np.float32) / 255.0
    low = np.asarray(lowpass(gen_img).convert("RGB").resize(gen_img.size), np.float32) / 255.0 \
        if lowpass is not None else gen
    lab_gen, lab_low, lab_ref = _lab(gen), _lab(low), _lab(ref)

    if mask is not None:
        m = cv2.resize(np.asarray(mask, np.uint8), gen_img.size, interpolation=cv2.INTER_NEAREST) > 0
        if m.sum() < 64:
            m = np.ones(m.shape, bool)
    else:
        m = np.ones(lab_gen.shape[:2], bool)

    mu_r, sd_r = lab_ref[m].mean(axis=0), lab_ref[m].std(axis=0)
    mu_g, sd_g = lab_low[m].mean(axis=0), lab_low[m].std(axis=0)
    ratio = np.clip(sd_r / (sd_g + eps), 0.5, 2.0)
    harm = (lab_gen - mu_g) * ratio + mu_r
    out = (1.0 - strength) * lab_gen + strength * harm
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    out[..., 1:] = np.clip(out[..., 1:], -127.0, 127.0)
    rgb = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)
    return Image.fromarray(_to_uint8(np.clip(rgb, 0.0, 1.0) * 255.0), "RGB")


def inpaint_foreground_region(img: np.ndarray, fg_mask: np.ndarray, work_dim: int = 512) -> np.ndarray:
    """Fill the foreground area with surrounding background colours (pre-blur)."""
    h, w = fg_mask.shape
    m = cv2.dilate(fg_mask.astype(np.uint8) * 255,
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if not m.any():
        return img
    sw, sh = fit_within(w, h, work_dim)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_m = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
    filled = cv2.inpaint(small, small_m, 5, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    sel = m > 0
    out[sel] = filled[sel]
    return out


def apply_layered_dof(background: Image.Image, depth_map: Optional[Image.Image],
                      alpha_full: Optional[np.ndarray], dof_strength: float) -> Image.Image:
    """Depth-layered pyramid bokeh (sharp / mid / far) with foreground occlusion fill."""
    r = float(dof_strength)
    if r <= 0 or depth_map is None:
        return background
    img = np.asarray(background.convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    if alpha_full is not None and (alpha_full > 128).any():
        img = inpaint_foreground_region(img, alpha_full > 128)
    depth = np.asarray(depth_map.convert("L").resize((w, h), Image.BILINEAR), np.float32)
    far = 1.0 - depth / 255.0
    far = cv2.GaussianBlur(far, (0, 0), 2.0)[..., None]
    src = img.astype(np.float32)
    mid = cv2.GaussianBlur(src, (0, 0), max(r / 2.0, 0.5))
    farb = cv2.GaussianBlur(src, (0, 0), max(r, 0.5))
    t1 = np.clip(far / 0.5, 0.0, 1.0)
    t2 = np.clip((far - 0.5) / 0.5, 0.0, 1.0)
    layer = src * (1.0 - t1) + mid * t1
    out = layer * (1.0 - t2) + farb * t2
    return Image.fromarray(_to_uint8(out), "RGB")


def build_exif(seed: Any, model_version: str, prompt: str, mode: str) -> Optional[bytes]:
    if piexif is None:
        return None
    try:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif["0th"][piexif.ImageIFD.Software] = f"SemanticCodec {VERSION}".encode("ascii")
        comment = json.dumps({"random_seed": seed, "model_version": model_version,
                              "prompt": prompt, "decode_mode": mode, "codec_version": VERSION})
        exif["Exif"][piexif.ExifIFD.UserComment] = comment.encode("utf-8")
        return piexif.dump(exif)
    except Exception:
        return None


# =============================================================================
# Decoder
# =============================================================================

@dataclass
class DecodeResult:
    image: Image.Image
    lite_image: Image.Image
    info: Dict[str, Any]

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.info.get("metrics", {})

    def to_jpeg_bytes(self, quality: int = 95) -> bytes:
        exif = self.info.get("_exif")
        kwargs: Dict[str, Any] = {"quality": quality}
        if exif:
            kwargs["exif"] = exif
        return _encode_image(self.image, "JPEG", **kwargs)

    def public_info(self) -> Dict[str, Any]:
        return {k: v for k, v in self.info.items() if not k.startswith("_")}


def _num_steps_for_effective(target: int, strength: float) -> Tuple[int, int]:
    """diffusers runs int(num_steps * strength) steps; pick num_steps so that the
    effective count equals ``target``. Returns (num_inference_steps, effective)."""
    strength = float(np.clip(strength, 0.05, 1.0))
    n = max(1, int(math.ceil(target / strength - 1e-6)))
    eff = min(int(n * strength), n)
    while eff < target:
        n += 1
        eff = min(int(n * strength), n)
    return n, max(1, eff)


def decode_semic(source: Union[str, bytes, io.IOBase, SemicPayload], mode: str = "Standard", *,
                 models: Optional[ModelManager] = None,
                 controlnet_scale: float = 0.5,
                 seg_scale: float = 0.4,
                 strength: Optional[float] = None,
                 guidance_scale: Optional[float] = None,
                 steps: Optional[int] = None,
                 dof_strength: float = 0.0,
                 color_harmonize: bool = True,
                 color_lock_strength: float = DEFAULT_COLOR_LOCK,
                 custom_prompt: Optional[str] = None,
                 custom_negative_prompt: Optional[str] = None,
                 seed: Optional[int] = None,
                 reference: Optional[Union[str, Image.Image]] = None,
                 progress: ProgressFn = None) -> DecodeResult:
    """Decode ``.semic.jpg`` into an image.

    ``progress(stage, step, total)`` is called with stage in
    {"loading", "generating", "postprocess", "done"}.
    """
    if mode not in DECODE_PROFILES:
        raise ValueError(f"Unknown decode mode '{mode}'. Choose from {list(DECODE_PROFILES)}")
    t0 = time.time()
    payload = source if isinstance(source, SemicPayload) else load_semic(source)
    manifest = payload.manifest
    w, h = payload.width, payload.height
    alpha_full = build_alpha_canvas(payload)
    archived_prompt = (manifest.get("prompt") or "").strip()
    custom_prompt_clean = (custom_prompt or "").strip()
    prompt = custom_prompt_clean or archived_prompt or DEFAULT_PROMPT
    prompt_overridden = bool(custom_prompt_clean and custom_prompt_clean != archived_prompt)
    negative = (custom_negative_prompt or "").strip() or manifest.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
    seed = int(seed if seed is not None else manifest.get("random_seed", 0)) % (2 ** 32)
    warnings: List[str] = list(payload.warnings)

    lite_image = decode_lite(payload)
    info: Dict[str, Any] = {
        "mode_requested": mode, "mode_used": "Lite", "steps": 0, "num_inference_steps": 0,
        "scheduler": None, "prompt": prompt, "prompt_overridden": prompt_overridden,
        "seed": seed, "width": w, "height": h, "source_version": payload.version,
        "color_harmonize": False, "dof_strength": float(dof_strength), "warnings": warnings,
    }

    result_img = lite_image
    if mode != "Lite":
        if models is None:
            warnings.append("No model manager supplied: decoded in Lite mode")
        else:
            _notify(progress, "loading", 0, 1)
            with models.lock:
                if not models.load_decoder_models():
                    warnings.append(f"Decoder unavailable ({models.decoder_error}); decoded in Lite mode")
                else:
                    result_img = _generate_background(
                        payload, mode, models, info, warnings, alpha_full, prompt, negative, seed,
                        controlnet_scale, seg_scale, strength, guidance_scale, steps,
                        dof_strength, color_harmonize, color_lock_strength, progress)

    info["decode_time_sec"] = round(time.time() - t0, 2)

    metrics: Dict[str, Any] = {
        "bpp": round(compute_bpp(payload.total_size, w, h), 4),
        "fg_area_ratio_pct": round(compute_fg_area_ratio_pct(alpha_full), 2),
    }
    if info["mode_used"] != "Lite":
        metrics["bg_color_fidelity"] = histogram_correlation(
            payload.thumb, result_img, mask=(255 - alpha_full) > 128)
    if reference is not None:
        ref_img, _ = load_image(reference)
        metrics.update(compute_reconstruction_metrics(result_img, ref_img, alpha_full))
        metrics["metric_reference"] = "original"
    elif payload.has_foreground:
        metrics["fg_psnr_db"] = compute_psnr(result_img, lite_image, alpha_full > 128)
        metrics["metric_reference"] = "lite"
    info["metrics"] = metrics
    info["_exif"] = build_exif(seed, manifest.get("model_version", ""), prompt, info["mode_used"])
    _notify(progress, "done", 1, 1)
    return DecodeResult(image=result_img, lite_image=lite_image, info=info)


def _generate_background(payload: SemicPayload, mode: str, models: ModelManager,
                         info: Dict[str, Any], warnings: List[str], alpha_full: np.ndarray,
                         prompt: str, negative: str, seed: int,
                         controlnet_scale: float, seg_scale: float,
                         strength: Optional[float], guidance_scale: Optional[float],
                         steps: Optional[int], dof_strength: float,
                         color_harmonize: bool, color_lock_strength: float,
                         progress: ProgressFn) -> Image.Image:
    prof = DECODE_PROFILES[mode]
    w, h = payload.width, payload.height
    manifest = payload.manifest

    cfg = models.configure_generation(
        use_lcm=prof.get("use_lcm", False),
        user_lora=manifest.get("lora_model", ""),
        user_lora_scale=float(manifest.get("lora_scale", 1.0)))
    warnings.extend(cfg["warnings"])
    try:
        strength_v = float(strength) if strength is not None else prof["strength"]
        strength_v = float(np.clip(strength_v, 0.05, 1.0))
        if prof.get("use_lcm") and not cfg["lcm"]:
            target_steps = int(steps) if steps else prof["fallback_steps"]
            gs = float(guidance_scale) if guidance_scale is not None else prof["fallback_guidance_scale"]
            if gs < 3.0:
                gs = prof["fallback_guidance_scale"]
            scheduler_name = "UniPC (Turbo fallback)"
        else:
            target_steps = int(steps) if steps else prof["steps"]
            gs = float(guidance_scale) if guidance_scale is not None else prof["guidance_scale"]
            scheduler_name = "LCM" if cfg["lcm"] else "UniPC"
        num_steps, eff_steps = _num_steps_for_effective(max(1, target_steps), strength_v)

        # Inputs
        proc_w, proc_h = inference_size(w, h)
        base_bg = build_base_background(payload, blur_radius=0).convert("RGB")
        blurred_bg = base_bg.filter(ImageFilter.GaussianBlur(radius=BG_INPUT_BLUR_RADIUS)).convert("RGB")
        gen_image = blurred_bg.resize((proc_w, proc_h), Image.LANCZOS).convert("RGB")

        bg_mask = (255 - alpha_full).astype(np.uint8)
        bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        gen_mask = Image.fromarray(bg_mask, "L").resize((proc_w, proc_h), Image.NEAREST).convert("L")

        depth = payload.depth_map or _gradient_depth(w, h)
        depth_ctrl = depth.resize((proc_w, proc_h), Image.BICUBIC).convert("RGB")
        seg_ctrl = prepare_seg_control(payload.seg_map, (proc_w, proc_h))
        eff_seg = float(seg_scale)
        if seg_ctrl is None:
            seg_ctrl = Image.new("RGB", (proc_w, proc_h), tuple(int(v) for v in ADE20K_PALETTE[0]))
            eff_seg = 0.0
            if manifest.get("encoded_mode") != "Lite":
                warnings.append("No segmentation map in file: Seg ControlNet disabled")
        else:
            seg_ctrl = seg_ctrl.convert("RGB")
        eff_cn = min(float(controlnet_scale), 0.3) if mode == "Standard" else float(controlnet_scale)

        generator = torch.Generator(device="cpu").manual_seed(seed)  # GPU-independent noise
        gen_start = time.time()
        _notify(progress, "generating", 0, eff_steps)

        def on_step(pipe, step_index, timestep, callback_kwargs):
            _notify(progress, "generating", min(step_index + 1, eff_steps), eff_steps,
                    time.time() - gen_start)
            return callback_kwargs

        models._empty_cache()
        out_bg = models.sd_pipeline(
            prompt=prompt,
            negative_prompt=negative,
            image=gen_image,
            mask_image=gen_mask,
            control_image=[depth_ctrl, seg_ctrl],
            num_inference_steps=num_steps,
            strength=strength_v,
            guidance_scale=gs,
            controlnet_conditioning_scale=[eff_cn, eff_seg],
            generator=generator,
            callback_on_step_end=on_step,
        ).images[0]
    finally:
        models.release_generation(cfg)

    _notify(progress, "postprocess", eff_steps, eff_steps)
    out_bg = out_bg.resize((w, h), Image.LANCZOS)

    if color_harmonize and color_lock_strength > 0:
        thumb_size = payload.thumb.size

        def lowpass(img: Image.Image) -> Image.Image:
            return img.resize(thumb_size, Image.LANCZOS).resize((w, h), Image.LANCZOS) \
                .filter(ImageFilter.GaussianBlur(radius=BG_INPUT_BLUR_RADIUS))

        out_bg = harmonize_color_lab(out_bg, blurred_bg, mask=(255 - alpha_full) > 128,
                                     strength=color_lock_strength, lowpass=lowpass)
        info["color_harmonize"] = True

    if dof_strength > 0:
        out_bg = apply_layered_dof(out_bg, payload.depth_map, alpha_full, dof_strength)

    info.update({
        "mode_used": mode, "steps": eff_steps, "num_inference_steps": num_steps,
        "scheduler": scheduler_name, "guidance_scale": gs, "strength": strength_v,
        "controlnet_scale": eff_cn, "seg_scale": eff_seg, "lcm": cfg["lcm"],
        "inference_size": [proc_w, proc_h],
        "color_lock_strength": float(color_lock_strength) if color_harmonize else 0.0,
    })
    return composite_foreground(out_bg, payload)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_profiles(source: Union[str, bytes, Image.Image], *, models: Optional[ModelManager] = None,
                       use_ai: bool = True, modes: Optional[List[str]] = None,
                       fg_alpha: Optional[np.ndarray] = None, seg_ids: Optional[np.ndarray] = None,
                       progress: ProgressFn = None) -> List[Dict[str, Any]]:
    """Encode with every profile and Lite-decode for a rate / distortion table."""
    original, size = load_image(source)
    if isinstance(source, (bytes, bytearray)):
        size = len(source)
    elif isinstance(source, str):
        size = os.path.getsize(source)
    rows: List[Dict[str, Any]] = []
    for mode in modes or list(ENCODE_PROFILES):
        _notify(progress, mode)
        enc = encode_to_semic(original, mode, models=models, use_ai=use_ai, original_file_size=size,
                              fg_alpha=fg_alpha, seg_ids=seg_ids, seed=0)
        payload = load_semic(enc.data)
        lite = decode_lite(payload)
        rec = compute_reconstruction_metrics(lite, original, build_alpha_canvas(payload))
        rows.append({
            "mode": mode,
            "semic_size": enc.stats["semic_size"],
            "bpp": enc.stats["bpp"],
            "compression_ratio_pct": enc.stats["compression_ratio_pct"],
            "ratio_display": enc.stats["ratio_display"],
            "fg_area_ratio_pct": enc.stats["fg_area_ratio_pct"],
            "components": enc.stats["components"],
            "lite_psnr_db": rec["psnr_db"],
            "lite_ssim": rec["ssim"],
            "fg_psnr_db": rec["fg_psnr_db"],
            "encode_time_sec": enc.stats["encode_time_sec"],
        })
    return rows
