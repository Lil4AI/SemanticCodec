#!/usr/bin/env python3
"""Semantic Codec v1.0.0 - CLI Mode

Usage:
    # 1. Encode
    python cli.py encode input.jpg [--mode Lite|Standard|HighFi] [-o output.semic.jpg] [--lora <file>] [--lora-scale 1.0]

    # 2. Decode
    python cli.py decode input.semic.jpg [--mode Lite|Turbo|Standard|HighFi] [-o decoded.jpg]
                                         [--cn-scale 0.5] [--seg-scale 0.4] [--strength 0.85]
                                         [--guidance 7.5] [--steps 20] [--dof 0]
                                         [--no-color-harmonize] [--prompt "custom prompt"]

    # 3. Inspect (.semic.jpg metadata, components, and bpp)
    python cli.py inspect input.semic.jpg

    # 4. Benchmark (Rate-Distortion evaluation across Lite, Standard, HighFi)
    python cli.py benchmark input.jpg
"""

import argparse
import json
import os
import sys
import time

import codec_engine as engine


def format_bytes(n):
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def cli_encode(args):
    """Encode an image into .semic.jpg format."""
    print(f"=== SemanticCodec v{engine.VERSION} [Encode] ===")
    print(f"Input : {args.input}")
    print(f"Mode  : {args.mode}")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    models = engine.ModelManager() if not args.no_ai else None
    t0 = time.time()

    def progress_cb(stage, step, total):
        print(f"  [{step}/{total}] {stage.capitalize()}...")

    result = engine.encode_to_semic(
        args.input,
        mode=args.mode,
        models=models,
        use_ai=not args.no_ai,
        lora_model=args.lora or "",
        lora_scale=args.lora_scale,
        progress=progress_cb
    )

    out_path = args.output or (os.path.splitext(args.input)[0] + ".semic.jpg")
    with open(out_path, "wb") as f:
        f.write(result.data)

    stats = result.stats
    print(f"\n[OK] Encoded successfully in {time.time() - t0:.2f}s!")
    print(f"  Output       : {out_path}")
    print(f"  Original Size: {format_bytes(stats.get('original_size'))}")
    print(f"  .semic Size  : {format_bytes(stats.get('semic_size'))}")
    print(f"  Bitrate (bpp): {stats.get('bpp')} bpp")
    print(f"  Compression  : {stats.get('compression_ratio_pct')}% reduction ({stats.get('ratio_display')})")
    print(f"  FG Area Ratio: {stats.get('fg_area_ratio_pct')}%")
    print(f"  Detected Style: {result.manifest.get('image_style')}")
    print(f"  Prompt       : {result.manifest.get('prompt')[:100]}...")


def cli_decode(args):
    """Decode a .semic.jpg file back to an image."""
    print(f"=== SemanticCodec v{engine.VERSION} [Decode] ===")
    print(f"Input : {args.input}")
    print(f"Mode  : {args.mode}")

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    models = engine.ModelManager() if args.mode != "Lite" else None
    t0 = time.time()

    def progress_cb(stage, step=0, total=1, elapsed=None):
        if stage == "generating":
            bar_len = 25
            filled = int(bar_len * step / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            eta = f" | ETA: {elapsed / step * (total - step):.1f}s" if (step > 0 and elapsed) else ""
            sys.stdout.write(f"\r  [{bar}] {step}/{total} Steps{eta} ")
            sys.stdout.flush()
        else:
            print(f"\n  Stage: {stage}...")

    result = engine.decode_semic(
        args.input,
        mode=args.mode,
        models=models,
        controlnet_scale=args.cn_scale,
        seg_scale=args.seg_scale,
        strength=args.strength,
        guidance_scale=args.guidance,
        steps=args.steps if args.steps > 0 else None,
        dof_strength=args.dof,
        color_harmonize=not args.no_color_harmonize,
        custom_prompt=args.prompt,
        progress=progress_cb
    )

    out_path = args.output or (os.path.splitext(args.input)[0] + f"_decoded_{args.mode.lower()}.jpg")
    with open(out_path, "wb") as f:
        f.write(result.to_jpeg_bytes())

    print(f"\n\n[OK] Decoded successfully in {time.time() - t0:.2f}s!")
    print(f"  Output       : {out_path}")
    print(f"  Mode Used    : {result.info.get('mode_used')}")
    print(f"  Inference    : {result.info.get('steps')} steps ({result.info.get('scheduler')})")
    metrics = result.info.get("metrics", {})
    if "bg_color_fidelity" in metrics and metrics["bg_color_fidelity"] is not None:
        print(f"  Color Fidelity: {metrics['bg_color_fidelity']}%")
    if "fg_psnr_db" in metrics and metrics["fg_psnr_db"] is not None:
        print(f"  FG PSNR      : {metrics['fg_psnr_db']} dB")


def cli_inspect(args):
    """Inspect .semic.jpg container metadata and breakdown."""
    print(f"=== SemanticCodec v{engine.VERSION} [Inspect] ===")
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    info = engine.inspect_semic(args.input)
    print(f"File         : {args.input}")
    print(f"Format       : Polyglot (.semic.jpg)")
    print(f"Codec Version: v{info.get('version')}")
    print(f"Dimensions   : {info.get('width')} x {info.get('height')}")
    print(f"Total Size   : {format_bytes(info.get('total_size'))} ({info.get('total_size')} bytes)")
    print(f"Bitrate (bpp): {info.get('bpp')} bpp")
    print(f"Guide Type   : {info.get('guide_type')}")
    print(f"Encoded Mode : {info.get('encoded_mode')}")
    print(f"Has FG       : {info.get('has_foreground')} (BBox: {info.get('fg_bbox')})")
    print(f"FG Area Ratio: {info.get('fg_area_ratio_pct')}%")
    print(f"Style        : {info.get('image_style')}")
    print(f"Seed         : {info.get('random_seed')}")
    print(f"Seg Format   : {info.get('seg_format')}")
    print(f"Prompt       : {info.get('prompt')}")
    print(f"Neg Prompt   : {info.get('negative_prompt')}")

    print("\n--- Component Breakdown ---")
    comps = info.get("components", {})
    total = info.get("total_size", 1)
    for name, size in sorted(comps.items(), key=lambda x: x[1], reverse=True):
        pct = (size / total) * 100
        print(f"  {name:<20}: {format_bytes(size):>10} ({pct:5.1f}%)")

    if info.get("warnings"):
        print("\nWarnings:")
        for w in info["warnings"]:
            print(f"  [!] {w}")


def cli_benchmark(args):
    """Run benchmark across Lite, Standard, HighFi profiles."""
    print(f"=== SemanticCodec v{engine.VERSION} [Benchmark] ===")
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running rate-distortion benchmark on: {args.input}")
    models = engine.ModelManager() if not args.no_ai else None

    def prog(mode):
        print(f"  Evaluating profile: {mode}...")

    rows = engine.benchmark_profiles(args.input, models=models, use_ai=not args.no_ai, progress=prog)

    print("\n" + "=" * 90)
    print(f"{'Profile':<10} | {'Size':<10} | {'bpp':<8} | {'Compression':<12} | {'FG Ratio':<9} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'Time (s)':<8}")
    print("-" * 90)
    for r in rows:
        psnr_str = f"{r['lite_psnr_db']:.2f}" if r['lite_psnr_db'] is not None else "N/A"
        ssim_str = f"{r['lite_ssim']:.4f}" if r['lite_ssim'] is not None else "N/A"
        print(f"{r['mode']:<10} | {format_bytes(r['semic_size']):<10} | {r['bpp']:<8.4f} | {r['compression_ratio_pct']:>5.1f}% ({r['ratio_display']}) | {r['fg_area_ratio_pct']:>6.1f}%  | {psnr_str:<9} | {ssim_str:<7} | {r['encode_time_sec']:<8.2f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description=f"Semantic Codec v{engine.VERSION} - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Encode
    enc = subparsers.add_parser("encode", help="Encode image to .semic.jpg")
    enc.add_argument("input", help="Input image path (JPEG, PNG, WebP)")
    enc.add_argument("--mode", choices=["Lite", "Standard", "HighFi"], default="Standard", help="Encode profile")
    enc.add_argument("-o", "--output", help="Output .semic.jpg path")
    enc.add_argument("--lora", help="Optional LoRA filename (in loras/)")
    enc.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale (default: 1.0)")
    enc.add_argument("--no-ai", action="store_true", help="Skip heavy AI (use GrabCut & gradient depth)")

    # 2. Decode
    dec = subparsers.add_parser("decode", help="Decode .semic.jpg to image")
    dec.add_argument("input", help="Input .semic.jpg path")
    dec.add_argument("--mode", choices=["Lite", "Turbo", "Standard", "HighFi"], default="Standard", help="Reconstruction profile")
    dec.add_argument("-o", "--output", help="Output image JPEG path")
    dec.add_argument("--cn-scale", type=float, default=0.5, help="ControlNet Depth scale (0.0 - 1.0)")
    dec.add_argument("--seg-scale", type=float, default=0.4, help="ControlNet Seg scale (0.0 - 1.0)")
    dec.add_argument("--strength", type=float, default=engine.DEFAULT_STRENGTH, help="Denoising strength (default: 0.85)")
    dec.add_argument("--guidance", type=float, default=None, help="Guidance scale (default: auto 1.8 for Turbo, 7.5 for Standard/HighFi)")
    dec.add_argument("--steps", type=int, default=0, help="Inference steps override (0 = auto for profile)")
    dec.add_argument("--dof", type=float, default=0.0, help="Depth of Field bokeh blur strength (0 = off)")
    dec.add_argument("--no-color-harmonize", action="store_true", help="Disable Reinhard LAB color harmonization")
    dec.add_argument("--prompt", help="Override background prompt during generation")

    # 3. Inspect
    insp = subparsers.add_parser("inspect", help="Inspect .semic.jpg metadata and components")
    insp.add_argument("input", help="Input .semic.jpg path")

    # 4. Benchmark
    bm = subparsers.add_parser("benchmark", help="Benchmark Lite, Standard, HighFi profiles on an image")
    bm.add_argument("input", help="Input image path")
    bm.add_argument("--no-ai", action="store_true", help="Use fallback GrabCut/gradient for faster offline testing")

    args = parser.parse_args()

    if args.command == "encode":
        cli_encode(args)
    elif args.command == "decode":
        cli_decode(args)
    elif args.command == "inspect":
        cli_inspect(args)
    elif args.command == "benchmark":
        cli_benchmark(args)


if __name__ == "__main__":
    main()
