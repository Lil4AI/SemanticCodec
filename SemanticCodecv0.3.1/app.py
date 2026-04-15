import os
import json
import zipfile
import io
import time
import random
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import piexif
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline as hf_pipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from rembg import remove

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp_encode'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# --- Global Models ---
models_loaded = False
blip_processor = None
blip_model = None
sd_pipeline = None
depth_estimator = None

def init_models():
    global models_loaded, blip_processor, blip_model, sd_pipeline, depth_estimator
    if models_loaded: return

    print("Loading AI Models... This might take a while on first run.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 1. BLIP for captioning (VQA-style prefix-based extraction)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        # 2. Depth Anything v2 Small for depth map estimation
        depth_estimator = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if device == "cuda" else -1
        )
        print("Depth Anything v2 Small loaded.")

        # 3. ControlNet Depth + SD Inpainting setup
        if device == "cuda":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16
            )
            sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            )
            sd_pipeline.scheduler = UniPCMultistepScheduler.from_config(sd_pipeline.scheduler.config)
            sd_pipeline.enable_model_cpu_offload()  # VRAM optimization
        else:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth"
            )
            sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=controlnet,
                safety_checker=None
            )
            sd_pipeline.to("cpu")

        models_loaded = True
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Allow server to start without models for testing the UI


def generate_depth_map(image):
    """Generate a depth map using Depth Anything v2 Small.
    Returns a PIL Image (grayscale) of the same size as input."""
    if depth_estimator is None:
        # Fallback: return a uniform gray image (no depth info)
        return Image.new("L", image.size, 128)

    result = depth_estimator(image)
    depth_map = result["depth"]  # PIL Image

    # Ensure it matches original image size
    if depth_map.size != image.size:
        depth_map = depth_map.resize(image.size, Image.LANCZOS)

    # Convert to grayscale if not already
    depth_map = depth_map.convert("L")
    return depth_map


def generate_vqa_caption(image, bg_mask_img):
    """VQA-style multi-angle prompt extraction.
    Uses foreground-masked image and multiple BLIP prefixes to extract
    rich background descriptions from multiple perspectives.

    Args:
        image: Original PIL Image (RGB)
        bg_mask_img: PIL Image (L) where 255 = background, 0 = foreground
    """
    if not models_loaded:
        return "A photo of a natural background environment, masterpiece, photorealistic, highly detailed, 8k resolution, sharp focus"

    # 1. Create foreground-masked image (foreground blacked out)
    #    This prevents BLIP from describing the subject (e.g. "a horse")
    bg_only = Image.composite(
        image,
        Image.new("RGB", image.size, (0, 0, 0)),
        bg_mask_img
    )

    # 2. Multi-angle prefix-based captioning
    prefixes = [
        "the background scene shows",       # General background description
        "the weather and lighting is",       # Atmosphere / lighting
        "the ground or surface is",          # Terrain / texture
        "in the distance there is",          # Far elements
    ]

    captions = []
    for prefix in prefixes:
        inputs = blip_processor(bg_only, text=prefix, return_tensors="pt")
        inputs = inputs.to(blip_model.device, blip_model.dtype)
        out = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption.strip())

    # 3. Combine all perspectives with comma separation
    combined = ", ".join(captions)

    # 4. Append quality enhancement prompts
    enhanced_prompt = combined + ", masterpiece, photorealistic, highly detailed, 8k resolution, sharp focus"

    print(f"[VQA Prompt] {enhanced_prompt}")
    return enhanced_prompt


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/encode', methods=['POST'])
def encode_image():
    init_models()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'Standard')
    lora_model = request.form.get('lora_model', '')
    lora_scale = float(request.form.get('lora_scale', 1.0))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load Original Image
        original_img = Image.open(file).convert("RGB")
        width, height = original_img.size

        # Measure original file size
        file.seek(0, 2)  # Seek to end
        original_file_size = file.tell()
        file.seek(0)

        # 1. Background Separation using rembg
        fg_rgba = remove(original_img)

        # Split into RGB and Alpha
        fg_color = fg_rgba.convert("RGB")
        fg_alpha = fg_rgba.split()[3]

        # Invert alpha to get background mask (255 where background is, 0 where object is)
        bg_mask = ImageOps.invert(fg_alpha)

        # 2. VQA-style multi-angle caption extraction (with foreground masked out)
        caption = generate_vqa_caption(original_img, bg_mask)

        # 3. Generate Depth Map (Depth Anything v2)
        depth_map = generate_depth_map(original_img)
        # Optimization (v0.3.1): Resize depth map for storage efficiency
        # 1024px is plenty for ControlNet Depth
        depth_map.thumbnail((1024, 1024), Image.LANCZOS)

        # 4. Generate Thumb (256x256, Quality 8)
        thumb = original_img.copy()
        thumb.thumbnail((256, 256))

        # 5. Save components to memory buffers
        fg_color_io = io.BytesIO()
        fg_color.save(fg_color_io, format='WEBP', quality=85, method=6)  # Lossy
        fg_color_size = fg_color_io.tell()

        fg_alpha_io = io.BytesIO()
        fg_alpha.save(fg_alpha_io, format='WEBP', lossless=True)  # Lossless
        fg_alpha_size = fg_alpha_io.tell()

        thumb_io = io.BytesIO()
        thumb.save(thumb_io, format='JPEG', quality=8)
        thumb_size = thumb_io.tell()

        depth_io = io.BytesIO()
        # Optimization (v0.3.1): Switch to WebP Lossless for depth map
        depth_map.save(depth_io, format='WEBP', lossless=True)
        depth_size = depth_io.tell()

        # 6. Create Manifest
        seed = random.randint(0, 2**32 - 1)
        manifest = {
            "version": "0.3.1",
            "guide_type": "depth",
            "width": width,
            "height": height,
            "prompt": caption,
            "random_seed": seed,
            "model_version": "stable-diffusion-inpainting-1.5_controlnet-depth",
            "encoded_mode": mode,
            "lora_model": lora_model,
            "lora_scale": lora_scale
        }

        # 7. Package as Polyglot (.semic.jpg): JPEG preview + ZIP payload
        final_io = io.BytesIO()

        # 7.1 Write Preview JPEG
        preview = Image.new("RGB", (width, height), (20, 20, 20))
        preview.paste(fg_color, (0, 0), fg_alpha)
        preview.thumbnail((512, 512))  # Lightweight preview
        preview.save(final_io, format='JPEG', quality=85)

        # 7.2 Append ZIP Data to the same buffer
        with zipfile.ZipFile(final_io, 'a', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('manifest.json', json.dumps(manifest))
            zipf.writestr('fg_color.webp', fg_color_io.getvalue())
            zipf.writestr('fg_alpha.webp', fg_alpha_io.getvalue())
            zipf.writestr('thumb.jpg', thumb_io.getvalue())
            zipf.writestr('depth_map.webp', depth_io.getvalue())

        semic_size = final_io.tell()
        final_io.seek(0)

        # 8. Compression statistics
        compression_ratio = (1 - semic_size / original_file_size) * 100 if original_file_size > 0 else 0

        # Save the encoded file temporarily for download
        temp_id = f"encoded_{int(time.time())}_{random.randint(1000,9999)}"
        temp_path = os.path.join(app.config['TEMP_FOLDER'], f"{temp_id}.semic.jpg")
        with open(temp_path, 'wb') as f:
            f.write(final_io.getvalue())

        return jsonify({
            'success': True,
            'download_id': temp_id,
            'filename': f'encoded_{int(time.time())}.semic.jpg',
            'stats': {
                'original_size': original_file_size,
                'semic_size': semic_size,
                'compression_ratio': round(compression_ratio, 1),
                'ratio_display': f"{original_file_size / semic_size:.1f}:1" if semic_size > 0 else "N/A",
                'components': {
                    'fg_color_webp': fg_color_size,
                    'fg_alpha_webp': fg_alpha_size,
                    'thumb_jpg': thumb_size,
                    'depth_map_webp': depth_size
                }
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/encode/download/<download_id>', methods=['GET'])
def download_encoded(download_id):
    """Download the encoded .semic.jpg file by its temporary ID."""
    # Sanitize download_id to prevent path traversal
    safe_id = secure_filename(download_id)
    temp_path = os.path.join(app.config['TEMP_FOLDER'], f"{safe_id}.semic.jpg")

    if not os.path.exists(temp_path):
        return jsonify({'error': 'File not found or expired'}), 404

    return send_file(
        temp_path,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f'{safe_id}.semic.jpg'
    )


@app.route('/api/decode', methods=['POST'])
def decode_image():
    init_models()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    decode_mode = request.form.get('mode', 'Standard')

    # Advanced parameters (with sensible defaults)
    controlnet_scale = float(request.form.get('controlnet_scale', 0.5))
    strength = float(request.form.get('strength', 1.0))
    guidance_scale = float(request.form.get('guidance_scale', 7.5))
    custom_steps = request.form.get('steps', '')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Check if it's a .semic or .semic.jpg file
        if file.filename.endswith('.semic') or file.filename.endswith('.semic.jpg'):
            with zipfile.ZipFile(file, 'r') as zipf:
                manifest = json.loads(zipf.read('manifest.json'))
                fg_color = Image.open(io.BytesIO(zipf.read('fg_color.webp'))).convert("RGB")
                fg_alpha = Image.open(io.BytesIO(zipf.read('fg_alpha.webp'))).convert("L")
                thumb = Image.open(io.BytesIO(zipf.read('thumb.jpg'))).convert("RGB")

                # Load depth map
                # v0.3.1: Supports depth_map.webp
                # Legacy: Check for depth_map.png as well
                try:
                    depth_map_data = zipf.read('depth_map.webp')
                except KeyError:
                    depth_map_data = zipf.read('depth_map.png')
                
                depth_map = Image.open(io.BytesIO(depth_map_data)).convert("L")
        else:
            return jsonify({'error': 'Only .semic or .semic.jpg files are supported'}), 400

        width = manifest["width"]
        height = manifest["height"]
        prompt = manifest["prompt"]
        seed = manifest["random_seed"]

        # Reconstruct base image from thumb
        base_bg = thumb.resize((width, height), Image.LANCZOS)

        # The mask where we want AI to paint (background = 255, foreground = 0)
        bg_mask = ImageOps.invert(fg_alpha)

        # === BUG B FIX: Mask Dilation ===
        # Expand the inpaint region by a few pixels beyond the foreground edge
        # to eliminate mosaic/blur artifacts at the boundary
        bg_mask_np = np.array(bg_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bg_mask_np = cv2.dilate(bg_mask_np, kernel, iterations=2)
        bg_mask = Image.fromarray(bg_mask_np)

        # Determine steps based on mode (can be overridden by advanced settings)
        if custom_steps:
            steps = int(custom_steps)
        elif decode_mode == 'HighFi':
            steps = 50
        elif decode_mode == 'Standard':
            steps = 20
        else:
            steps = 20

        # If Lite mode: AI generation skipped. Use scaled thumb as background.
        if decode_mode == 'Lite' or not models_loaded:
            result_img = base_bg.copy()
            # Paste foreground
            result_img.paste(fg_color, (0, 0), fg_alpha)

        else:
            # Prepare AI generation
            generator = torch.Generator(
                device="cpu" if not torch.cuda.is_available() else "cuda"
            ).manual_seed(seed)

            # Upscale depth map to correct dims for ControlNet
            control_image = depth_map.resize((width, height), Image.LANCZOS)

            # For Standard mode, reduce ControlNet influence
            if decode_mode == 'Standard':
                effective_cn_scale = min(controlnet_scale, 0.3)
            else:
                effective_cn_scale = controlnet_scale

            # Resize constraints (SD likes multiples of 8)
            proc_w = (width // 8) * 8
            proc_h = (height // 8) * 8

            # Since VRAM is limited, scale down during inference if too large
            if proc_w * proc_h > 1024 * 1024:  # max 1MP limit for basic VRAM
                scale = (1024 * 1024) ** 0.5 / (proc_w * proc_h) ** 0.5
                proc_w = int((proc_w * scale) // 8) * 8
                proc_h = int((proc_h * scale) // 8) * 8

            # Blur the thumbnail to dissolve JPEG block noise
            gen_bg = base_bg.filter(ImageFilter.GaussianBlur(radius=15)).resize(
                (proc_w, proc_h), Image.LANCZOS
            )
            gen_mask = bg_mask.resize((proc_w, proc_h), Image.LANCZOS)

            # Convert depth control image to RGB (ControlNet Depth expects RGB)
            gen_control = control_image.resize((proc_w, proc_h), Image.LANCZOS)
            gen_control_rgb = Image.merge("RGB", [gen_control, gen_control, gen_control])

            # LoRA Handling
            lora_model = manifest.get("lora_model", "")
            lora_scale_val = float(manifest.get("lora_scale", 1.0))
            if lora_model and os.path.exists(os.path.join("loras", lora_model)):
                try:
                    sd_pipeline.load_lora_weights("loras", weight_name=lora_model)
                    print(f"Loaded LoRA: {lora_model} at scale {lora_scale_val}")
                except Exception as e:
                    print(f"Warning: Failed to load LoRA {lora_model}: {e}")
                    lora_model = ""
            elif lora_model:
                print(f"Warning: LoRA `{lora_model}` not found in loras/ directory. Falling back to base model.")
                lora_model = ""

            # Pipeline kwargs
            kwargs = {}
            if lora_model:
                kwargs["cross_attention_kwargs"] = {"scale": lora_scale_val}

            # Run SD Pipeline
            out_bg = sd_pipeline(
                prompt=prompt,
                negative_prompt="blurry, painting, abstract, artificial, low quality, deformed",
                image=gen_bg,
                mask_image=gen_mask,
                control_image=gen_control_rgb,
                num_inference_steps=steps,
                strength=strength,
                generator=generator,
                controlnet_conditioning_scale=effective_cn_scale,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]

            # Unload LoRA to keep base model clean for next request
            if lora_model:
                sd_pipeline.unload_lora_weights()

            # Resize back to original
            out_bg = out_bg.resize((width, height), Image.LANCZOS)

            # Paste true foreground (Lossless WebP = high quality)
            result_img = out_bg.copy()
            result_img.paste(fg_color, (0, 0), fg_alpha)

        # Add Exif Data to prevent Semantic Drift
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        manifest_str = json.dumps({
            "random_seed": seed,
            "model_version": manifest["model_version"],
            "prompt": prompt
        })
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = manifest_str.encode("utf-8")
        exif_bytes = piexif.dump(exif_dict)

        # Save to IO
        out_io = io.BytesIO()
        result_img.save(out_io, format='JPEG', quality=95, exif=exif_bytes)
        out_io.seek(0)

        return send_file(
            out_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'decoded_{decode_mode}.jpg'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
