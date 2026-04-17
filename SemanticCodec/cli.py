#!/usr/bin/env python3
"""Semantic Codec v0.4.1 - CLI Mode

Usage:
    python cli.py encode <image_path> [--mode Standard|HighFi|Lite] [--output <path>]
    python cli.py decode <semic_path> [--mode Standard|HighFi|Lite] [--output <path>]
                                      [--cn-scale 0.5] [--seg-scale 0.4]
                                      [--strength 1.0] [--guidance 7.5]
                                      [--steps 20] [--dof 0]
"""

import argparse
import sys
import os
import json
import zipfile
import io
import time
import random

from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import piexif

# Import shared functions from app.py
# We need to handle the Flask imports gracefully
os.environ.setdefault('FLASK_SKIP_DOTENV', '1')
import app
from rembg import remove
import torch


def cli_encode(args):
    """Encode an image into .semic.jpg format."""
    app.init_models()

    print(f"Encoding: {args.input}")
    print(f"Mode: {args.mode}")

    # Load image
    original_img = Image.open(args.input).convert("RGB")
    width, height = original_img.size
    original_file_size = os.path.getsize(args.input)

    print(f"Image: {width}x{height} ({original_file_size / 1024:.1f} KB)")

    # 1. Background Separation
    print("[1/5] Separating foreground...")
    fg_rgba = remove(original_img)
    fg_color = fg_rgba.convert("RGB")
    fg_alpha = fg_rgba.split()[3]
    bg_mask = ImageOps.invert(fg_alpha)

    # 2. VQA Caption
    print("[2/5] Extracting VQA captions...")
    pos_prompt, neg_prompt, img_style = app.generate_vqa_caption(original_img, bg_mask)
    print(f"  Style: {img_style}")
    print(f"  Prompt: {pos_prompt[:80]}...")

    # 3. Depth Map
    print("[3/5] Generating depth map...")
    depth_map = app.generate_depth_map(original_img)
    depth_map.thumbnail((1024, 1024), Image.LANCZOS)

    # 3.5 Segmentation Map
    print("[3.5/5] Generating segmentation map...")
    seg_map = app.generate_seg_map(original_img)
    seg_map.thumbnail((1024, 1024), Image.LANCZOS)

    # 4. Thumbnail
    print("[4/5] Creating thumbnail...")
    thumb = original_img.copy()
    thumb.thumbnail((256, 256))

    # 5. Packaging
    print("[5/5] Packaging .semic.jpg...")

    fg_color_io = io.BytesIO()
    fg_color.save(fg_color_io, format='WEBP', quality=85, method=6)

    fg_alpha_io = io.BytesIO()
    fg_alpha.save(fg_alpha_io, format='WEBP', lossless=True)

    thumb_io = io.BytesIO()
    thumb.save(thumb_io, format='JPEG', quality=8)

    depth_io = io.BytesIO()
    depth_map.save(depth_io, format='WEBP', lossless=True)

    seg_io = io.BytesIO()
    seg_map.save(seg_io, format='WEBP', lossless=True)

    seed = random.randint(0, 2**32 - 1)
    manifest = {
        "version": "0.4.1",
        "guide_type": "depth",
        "width": width,
        "height": height,
        "prompt": pos_prompt,
        "negative_prompt": neg_prompt,
        "image_style": img_style,
        "random_seed": seed,
        "model_version": "stable-diffusion-inpainting-1.5_multi-controlnet-depth-seg",
        "encoded_mode": args.mode,
        "lora_model": "",
        "lora_scale": 1.0
    }

    final_io = io.BytesIO()

    # Preview JPEG
    preview = Image.new("RGB", (width, height), (20, 20, 20))
    preview.paste(fg_color, (0, 0), fg_alpha)
    preview.thumbnail((512, 512))
    preview.save(final_io, format='JPEG', quality=85)

    # ZIP payload
    with zipfile.ZipFile(final_io, 'a', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr('manifest.json', json.dumps(manifest))
        zipf.writestr('fg_color.webp', fg_color_io.getvalue())
        zipf.writestr('fg_alpha.webp', fg_alpha_io.getvalue())
        zipf.writestr('thumb.jpg', thumb_io.getvalue())
        zipf.writestr('depth_map.webp', depth_io.getvalue())
        zipf.writestr('seg_map.webp', seg_io.getvalue())

    semic_size = final_io.tell()
    final_io.seek(0)

    # Determine output path
    output_path = args.output or os.path.splitext(args.input)[0] + '.semic.jpg'
    with open(output_path, 'wb') as f:
        f.write(final_io.getvalue())

    ratio = (1 - semic_size / original_file_size) * 100 if original_file_size > 0 else 0
    print(f"\n✅ Encoded successfully!")
    print(f"  Output: {output_path}")
    print(f"  Original: {original_file_size / 1024:.1f} KB → .semic: {semic_size / 1024:.1f} KB")
    print(f"  Compression: {ratio:.1f}% reduced ({original_file_size / semic_size:.1f}:1)")


def cli_decode(args):
    """Decode a .semic.jpg file back to an image."""
    app.init_models()

    print(f"Decoding: {args.input}")
    print(f"Mode: {args.mode}")

    # Load .semic file
    with zipfile.ZipFile(args.input, 'r') as zipf:
        manifest = json.loads(zipf.read('manifest.json'))
        fg_color = Image.open(io.BytesIO(zipf.read('fg_color.webp'))).convert("RGB")
        fg_alpha = Image.open(io.BytesIO(zipf.read('fg_alpha.webp'))).convert("L")
        thumb = Image.open(io.BytesIO(zipf.read('thumb.jpg'))).convert("RGB")

        try:
            depth_map_data = zipf.read('depth_map.webp')
        except KeyError:
            depth_map_data = zipf.read('depth_map.png')
        depth_map = Image.open(io.BytesIO(depth_map_data)).convert("L")

        try:
            seg_map = Image.open(io.BytesIO(zipf.read('seg_map.webp'))).convert("RGB")
        except KeyError:
            seg_map = None

    width = manifest["width"]
    height = manifest["height"]
    prompt = manifest["prompt"]
    seed = manifest["random_seed"]

    print(f"  Dimensions: {width}x{height}")
    print(f"  Prompt: {prompt[:80]}...")

    base_bg = thumb.resize((width, height), Image.LANCZOS)
    bg_mask = ImageOps.invert(fg_alpha)

    # Mask dilation
    bg_mask_np = np.array(bg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_mask_np = cv2.dilate(bg_mask_np, kernel, iterations=2)
    bg_mask = Image.fromarray(bg_mask_np)

    steps = args.steps
    if steps == 0:
        steps = 50 if args.mode == 'HighFi' else 20

    if args.mode == 'Lite' or not app.models_loaded:
        print("  Mode: Lite (no AI generation)")
        result_img = base_bg.copy()
        result_img.paste(fg_color, (0, 0), fg_alpha)
    else:
        print(f"  Running SD pipeline ({steps} steps)...")
        generator = torch.Generator(
            device="cpu" if not torch.cuda.is_available() else "cuda"
        ).manual_seed(seed)

        control_image = depth_map.resize((width, height), Image.LANCZOS)

        effective_cn_scale = min(args.cn_scale, 0.3) if args.mode == 'Standard' else args.cn_scale

        proc_w = (width // 8) * 8
        proc_h = (height // 8) * 8
        max_pixels = 768 * 768
        if proc_w * proc_h > max_pixels:
            scale = max_pixels ** 0.5 / (proc_w * proc_h) ** 0.5
            proc_w = int((proc_w * scale) // 8) * 8
            proc_h = int((proc_h * scale) // 8) * 8

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gen_bg = base_bg.filter(ImageFilter.GaussianBlur(radius=15)).resize(
            (proc_w, proc_h), Image.LANCZOS
        )
        gen_mask = bg_mask.resize((proc_w, proc_h), Image.LANCZOS)

        gen_control_depth = control_image.resize((proc_w, proc_h), Image.LANCZOS)
        gen_control_depth_rgb = Image.merge("RGB", [gen_control_depth, gen_control_depth, gen_control_depth])

        if seg_map is not None:
            gen_control_seg = seg_map.resize((proc_w, proc_h), Image.LANCZOS)
        else:
            gen_control_seg = Image.new("RGB", (proc_w, proc_h), (120, 120, 120))

        gen_control_images = [gen_control_depth_rgb, gen_control_seg]

        # Progress callback for CLI
        start_time = time.time()

        def cli_progress(pipe, step_index, timestep, callback_kwargs):
            elapsed = time.time() - start_time
            steps_done = step_index + 1
            eta = (elapsed / steps_done) * (steps - steps_done) if steps_done > 0 else 0
            bar_len = 30
            filled = int(bar_len * steps_done / steps)
            bar = '█' * filled + '░' * (bar_len - filled)
            sys.stdout.write(f'\r  [{bar}] {steps_done}/{steps} | ETA: {eta:.1f}s ')
            sys.stdout.flush()
            return callback_kwargs

        neg = manifest.get("negative_prompt", "blurry, painting, abstract, artificial, low quality, deformed")

        out_bg = app.sd_pipeline(
            prompt=prompt,
            negative_prompt=neg,
            image=gen_bg,
            mask_image=gen_mask,
            control_image=gen_control_images,
            num_inference_steps=steps,
            strength=args.strength,
            generator=generator,
            controlnet_conditioning_scale=[effective_cn_scale, args.seg_scale],
            guidance_scale=args.guidance,
            callback_on_step_end=cli_progress,
        ).images[0]

        print()  # newline after progress bar

        out_bg = out_bg.resize((width, height), Image.LANCZOS)

        # DOF post-processing
        if args.dof > 0:
            print(f"  Applying DOF blur (strength={args.dof})...")
            out_bg_blurred = out_bg.filter(ImageFilter.GaussianBlur(radius=args.dof))
            depth_for_blend = depth_map.resize((width, height), Image.LANCZOS)
            inv_depth = ImageOps.invert(depth_for_blend)
            out_bg = Image.composite(out_bg_blurred, out_bg, inv_depth)

        result_img = out_bg.copy()
        result_img.paste(fg_color, (0, 0), fg_alpha)

    # Add EXIF
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    manifest_str = json.dumps({
        "random_seed": seed,
        "model_version": manifest["model_version"],
        "prompt": prompt
    })
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = manifest_str.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)

    output_path = args.output or os.path.splitext(args.input)[0] + f'_decoded_{args.mode}.jpg'
    result_img.save(output_path, format='JPEG', quality=95, exif=exif_bytes)

    print(f"\n✅ Decoded successfully!")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Codec v0.4.1 - CLI Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py encode photo.jpg
  python cli.py encode photo.jpg --mode HighFi --output compressed.semic.jpg
  python cli.py decode compressed.semic.jpg
  python cli.py decode compressed.semic.jpg --mode HighFi --steps 50 --dof 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Encode subcommand
    enc = subparsers.add_parser('encode', help='Encode image to .semic.jpg')
    enc.add_argument('input', help='Input image path (JPEG/PNG/WebP)')
    enc.add_argument('--mode', choices=['Lite', 'Standard', 'HighFi'], default='Standard')
    enc.add_argument('--output', '-o', help='Output .semic.jpg path')

    # Decode subcommand
    dec = subparsers.add_parser('decode', help='Decode .semic.jpg to image')
    dec.add_argument('input', help='Input .semic or .semic.jpg path')
    dec.add_argument('--mode', choices=['Lite', 'Standard', 'HighFi'], default='Standard')
    dec.add_argument('--output', '-o', help='Output JPEG path')
    dec.add_argument('--cn-scale', type=float, default=0.5, help='ControlNet Depth scale (0-1)')
    dec.add_argument('--seg-scale', type=float, default=0.4, help='ControlNet Seg scale (0-1)')
    dec.add_argument('--strength', type=float, default=1.0, help='Denoise strength (0.5-1)')
    dec.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (1-20)')
    dec.add_argument('--steps', type=int, default=0, help='Inference steps (0=auto)')
    dec.add_argument('--dof', type=float, default=0, help='DOF blur strength (0=off)')

    args = parser.parse_args()

    if args.command == 'encode':
        cli_encode(args)
    elif args.command == 'decode':
        cli_decode(args)


if __name__ == '__main__':
    main()
