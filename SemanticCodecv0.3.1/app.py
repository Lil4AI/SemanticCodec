import os
import json
import zipfile
import io
import time
import random
import threading
from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import piexif
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, SegformerImageProcessor, SegformerForSemanticSegmentation, pipeline as hf_pipeline
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from rembg import remove

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp_encode'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# --- SSE Progress State ---
decode_progress = {}  # session_id -> {step, total, eta_seconds, start_time}
progress_lock = threading.Lock()

# --- Global Models ---
models_loaded = False
blip_processor = None
blip_model = None
sd_pipeline = None
depth_estimator = None
seg_feature_extractor = None
seg_model = None

# ADE20K palette (150 classes) for ControlNet Seg
# This is the standard color palette used by control_v11p_sd15_seg
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

def init_models():
    global models_loaded, blip_processor, blip_model, sd_pipeline, depth_estimator, seg_feature_extractor, seg_model
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

        # 3. Segformer for semantic segmentation (ADE20K 150 classes)
        seg_feature_extractor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("Segformer-B0 (ADE20K) loaded.")

        if device == "cuda":
            print(f"CUDA Device Detected: {torch.cuda.get_device_name(0)}")
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total VRAM: {vram_total:.2f} GB")

            # 4. Multi-ControlNet (Depth + Seg) + SD Inpainting setup
            controlnet_depth = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16
            )
            controlnet_seg = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_seg",
                torch_dtype=torch.float16
            )
            print("Multi-ControlNet (Depth + Seg) loaded in FP16.")

            sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=[controlnet_depth, controlnet_seg],
                torch_dtype=torch.float16,
                safety_checker=None
            )
            sd_pipeline.scheduler = UniPCMultistepScheduler.from_config(sd_pipeline.scheduler.config)
            
            # Optimization for 8GB VRAM (RTX 2060 Super)
            # 8GB is actually easily overwhelmed by Multi-ControlNet + SD1.5 at high resolutions.
            # We must use model_cpu_offload to prevent WDDM from swapping to system RAM (which causes 16s/it).
            sd_pipeline.enable_model_cpu_offload()
            try:
                sd_pipeline.enable_xformers_memory_efficient_attention()
                print("xformers enabled for VRAM optimization.")
            except Exception:
                sd_pipeline.enable_attention_slicing()
                print("Attention slicing enabled (xformers not found).")
        else:
            print("Running on CPU. This will be slow!")
            # 4. Multi-ControlNet (Depth + Seg) + SD Inpainting setup
            controlnet_depth = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth"
            )
            controlnet_seg = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_seg"
            )
            sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=[controlnet_depth, controlnet_seg],
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


def generate_seg_map(image):
    """Generate a semantic segmentation map using Segformer (ADE20K 150 classes).
    Returns an RGB PIL Image using the ADE20K color palette, suitable for ControlNet Seg."""
    if seg_model is None or seg_feature_extractor is None:
        # Fallback: return a uniform gray image
        return Image.new("RGB", image.size, (120, 120, 120))

    inputs = seg_feature_extractor(images=image, return_tensors="pt")
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    inputs = {k: v.to(seg_model.device, model_dtype) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = seg_model(**inputs)

    # Upsample logits to original image size
    logits = torch.nn.functional.interpolate(
        outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map_np = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Convert class indices to ADE20K color palette
    color_seg = ADE20K_PALETTE[seg_map_np]
    seg_img = Image.fromarray(color_seg)
    return seg_img


def generate_vqa_caption(image, bg_mask_img):
    """VQA-style multi-angle prompt extraction.
    Uses foreground-masked image and multiple BLIP prefixes to extract
    rich background descriptions from multiple perspectives.

    Args:
        image: Original PIL Image (RGB)
        bg_mask_img: PIL Image (L) where 255 = background, 0 = foreground
    """
    if not models_loaded:
        return "A photo of a natural background environment, masterpiece, photorealistic, highly detailed, 8k resolution, sharp focus", "blurry, low quality", "photo"

    # 1. Style Classification using BLIP (Zero-shot approach)
    # Stage 1: Descriptive prefix
    style_prefix = "The artistic style of this image is best described as a"
    inputs = blip_processor(image, text=style_prefix, return_tensors="pt").to(blip_model.device, blip_model.dtype)
    out = blip_model.generate(**inputs, max_new_tokens=20)
    raw_style = blip_processor.decode(out[0], skip_special_tokens=True).lower()

    def categorize(text):
        if any(w in text for w in ["photo", "photograph", "real", "camera", "picture", "captured", "dslr", "natural"]):
            return "photo"
        if any(w in text for w in ["illust", "anime", "cartoon", "drawing", "sketch", "2d", "comic", "manga", "digital art", "artwork", "vector"]):
            return "illustration"
        if any(w in text for w in ["paint", "watercolor", "oil", "acrylic", "canvas", "brush", "stroke"]):
            return "painting"
        if any(w in text for w in ["pattern", "texture", "flat", "abstract", "geometric", "design"]):
            return "pattern"
        return None

    detected_style = categorize(raw_style)

    # Stage 2: Fallback to direct Q&A if Stage 1 is inconclusive
    if not detected_style:
        qa_prompt = "Question: Is this image a photograph, a digital illustration, or a painting? Answer:"
        inputs = blip_processor(image, text=qa_prompt, return_tensors="pt").to(blip_model.device, blip_model.dtype)
        out = blip_model.generate(**inputs, max_new_tokens=10)
        qa_answer_raw = blip_processor.decode(out[0], skip_special_tokens=True).lower()
        # Remove the prompt text from the answer to prevent false keyword matches
        qa_answer = qa_answer_raw.replace(qa_prompt.lower(), "").strip()
        
        detected_style = categorize(qa_answer) or "photo" # default fallback
        print(f"[VQA Style Match] Stage 1 inconclusive. Stage 2 (Q&A): '{qa_answer}' -> Categorized: {detected_style}")
    else:
        print(f"[VQA Style Match] Raw: '{raw_style}' -> Categorized: {detected_style}")

    # 2. Create foreground-masked image (foreground blacked out)
    #    This prevents BLIP from describing the subject (e.g. "a horse")
    bg_only = Image.composite(
        image,
        Image.new("RGB", image.size, (0, 0, 0)),
        bg_mask_img
    )

    # 3. Multi-angle prefix-based captioning
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

    combined = ", ".join(captions)

    # 4. Construct Positive and Negative prompts based on categorized style
    if detected_style == "photo":
        raw_pos = combined + ", true photographic masterpiece, f/8, deep depth of field, sharp focus everywhere, photorealistic, 8k, highly detailed photography"
        neg_prompt = "blurry, shallow depth of field, bokeh, out of focus, painting, illustration, cartoon, fake, 3d render, abstract, oil painting, ugly, deformed, grain, noise"
    elif detected_style == "illustration":
        raw_pos = combined + ", highly detailed illustration, 2d anime style, sharp lines, vibrant colors, flat shading, masterpiece"
        neg_prompt = "photograph, realistic, 3d render, blurry, bokeh, painting, messy, distorted, photographic"
    elif detected_style == "painting":
        raw_pos = combined + ", beautiful painting, traditional art style, visible brushstrokes, canvas texture, masterpiece"
        neg_prompt = "photograph, realistic, 3d render, blurry, anime, cartoon, flat, ugly, photo"
    elif detected_style == "pattern":
        raw_pos = combined + ", seamless pattern, flat texture, 2d design, repeating elements, high quality graphic design"
        neg_prompt = "depth, 3d, realistic, blurry, bokeh, photograph, perspective, messy"
    else:
        raw_pos = combined + ", masterpiece, high quality"
        neg_prompt = "blurry, low quality"

    # Clean and truncate the positive prompt to stay under 77 token limit
    pos_prompt = clean_and_truncate_prompt(raw_pos)

    print(f"[VQA Prompt] {pos_prompt}\n[VQA Negative] {neg_prompt}")
    return pos_prompt, neg_prompt, detected_style


def clean_and_truncate_prompt(prompt, max_tokens=70):
    """Clean prompt of redundant commas and truncate keywords to stay under CLIP limits.
    max_tokens=70 (leaving some buffer for the 77 limit)."""
    # Simple heuristic: split by comma, remove empty, and take first N parts
    parts = [p.strip() for p in prompt.split(',') if p.strip()]
    
    # Estimate tokens (approx 1.3 tokens per word)
    # We take the most important parts (the first few are usually BLIP descriptions, then style)
    # We assume roughly 1 keyword per 2 tokens
    limited_parts = []
    current_count = 0
    for p in parts:
        word_count = len(p.split())
        estimated_tokens = word_count * 1.5 + 1 # rough overhead
        if current_count + estimated_tokens > max_tokens:
            break
        limited_parts.append(p)
        current_count += estimated_tokens
    
    return ", ".join(limited_parts)


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
        pos_prompt, neg_prompt, img_style = generate_vqa_caption(original_img, bg_mask)

        # 3. Generate Depth Map (Depth Anything v2)
        depth_map = generate_depth_map(original_img)
        # Optimization (v0.3.1): Resize depth map for storage efficiency
        # 1024px is plenty for ControlNet Depth
        depth_map.thumbnail((1024, 1024), Image.LANCZOS)

        # 3.5 Generate Segmentation Map (Segformer ADE20K)
        seg_map = generate_seg_map(original_img)
        seg_map.thumbnail((1024, 1024), Image.LANCZOS)

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

        seg_io = io.BytesIO()
        seg_map.save(seg_io, format='WEBP', lossless=True)
        seg_size = seg_io.tell()

        # 6. Create Manifest
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
            zipf.writestr('seg_map.webp', seg_io.getvalue())

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
                    'depth_map_webp': depth_size,
                    'seg_map_webp': seg_size
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
    session_id = request.form.get('session_id', f"dec_{int(time.time())}_{random.randint(1000,9999)}")

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

                # Load segmentation map (v0.4.0+)
                try:
                    seg_map_data = zipf.read('seg_map.webp')
                    seg_map = Image.open(io.BytesIO(seg_map_data)).convert("RGB")
                except KeyError:
                    seg_map = None  # Legacy files without seg map
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
            # Mark progress done immediately for Lite
            with progress_lock:
                decode_progress[session_id] = {'step': 1, 'total': 1, 'eta_seconds': 0, 'done': True}

        else:
            # Initialize progress tracking
            gen_start_time = time.time()
            with progress_lock:
                decode_progress[session_id] = {'step': 0, 'total': steps, 'eta_seconds': 0, 'done': False}

            def progress_callback(pipe, step_index, timestep, callback_kwargs):
                elapsed = time.time() - gen_start_time
                steps_done = step_index + 1
                eta = (elapsed / steps_done) * (steps - steps_done) if steps_done > 0 else 0
                with progress_lock:
                    decode_progress[session_id] = {
                        'step': steps_done, 'total': steps,
                        'eta_seconds': eta, 'done': False
                    }
                return callback_kwargs

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
            # 8GB VRAM with Multi-ControlNet maxes out safely around 768x768 (0.59 MP).
            max_pixels = 768 * 768
            if proc_w * proc_h > max_pixels:
                scale = max_pixels ** 0.5 / (proc_w * proc_h) ** 0.5
                proc_w = int((proc_w * scale) // 8) * 8
                proc_h = int((proc_h * scale) // 8) * 8

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear memory buildup to prevent WDDM swap

            # Blur the thumbnail to dissolve JPEG block noise
            gen_bg = base_bg.filter(ImageFilter.GaussianBlur(radius=15)).resize(
                (proc_w, proc_h), Image.LANCZOS
            )
            gen_mask = bg_mask.resize((proc_w, proc_h), Image.LANCZOS)

            # Convert depth control image to RGB (ControlNet Depth expects RGB)
            gen_control_depth = control_image.resize((proc_w, proc_h), Image.LANCZOS)
            gen_control_depth_rgb = Image.merge("RGB", [gen_control_depth, gen_control_depth, gen_control_depth])

            # Prepare segmentation control image for Multi-ControlNet
            if seg_map is not None:
                gen_control_seg = seg_map.resize((proc_w, proc_h), Image.LANCZOS)
            else:
                # Fallback: uniform gray if no seg map in archive
                gen_control_seg = Image.new("RGB", (proc_w, proc_h), (120, 120, 120))

            # Multi-ControlNet: list of control images [depth, seg]
            gen_control_images = [gen_control_depth_rgb, gen_control_seg]

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

            # Run SD Pipeline (Multi-ControlNet: Depth + Seg)
            # controlnet_conditioning_scale is now a list: [depth_scale, seg_scale]
            seg_scale = float(request.form.get('seg_scale', 0.4))
            out_bg = sd_pipeline(
                prompt=prompt,
                negative_prompt=manifest.get("negative_prompt", "blurry, painting, abstract, artificial, low quality, deformed"),
                image=gen_bg,
                mask_image=gen_mask,
                control_image=gen_control_images,
                num_inference_steps=steps,
                strength=strength,
                generator=generator,
                controlnet_conditioning_scale=[effective_cn_scale, seg_scale],
                guidance_scale=guidance_scale,
                callback_on_step_end=progress_callback,
                **kwargs
            ).images[0]

            # Mark progress as done
            with progress_lock:
                decode_progress[session_id] = {'step': steps, 'total': steps, 'eta_seconds': 0, 'done': True}

            # Unload LoRA to keep base model clean for next request
            if lora_model:
                sd_pipeline.unload_lora_weights()

            # Resize back to original
            out_bg = out_bg.resize((width, height), Image.LANCZOS)
            
            # --- POST-PROCESSING: Artificial Depth of Field (DOF) ---
            # Apply blur to background elements based on depth map if user set dof_strength > 0
            dof_strength = float(request.form.get("dof_strength", 0.0))
            if dof_strength > 0:
                out_bg_blurred = out_bg.filter(ImageFilter.GaussianBlur(radius=dof_strength))
                # Depth map values: 255 = near (foreground), 0 = far (background)
                # We want blur on the background (far), so we invert the depth map
                # such that 255 means "far" and gets the blur mask.
                depth_for_blend = depth_map.resize((width, height), Image.LANCZOS)
                inv_depth = ImageOps.invert(depth_for_blend)
                out_bg = Image.composite(out_bg_blurred, out_bg, inv_depth)

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

        response = send_file(
            out_io,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'decoded_{decode_mode}.jpg'
        )
        response.headers['X-Session-Id'] = session_id
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- SSE Progress Endpoint ---
@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events endpoint for decode progress."""
    def generate():
        last_step = -1
        while True:
            with progress_lock:
                info = decode_progress.get(session_id, {})
            step = info.get('step', 0)
            total = info.get('total', 1)
            eta = info.get('eta_seconds', 0)
            done = info.get('done', False)

            if step != last_step or done:
                data = json.dumps({'step': step, 'total': total, 'eta_seconds': round(eta, 1), 'done': done})
                yield f"data: {data}\n\n"
                last_step = step

            if done:
                # Clean up
                with progress_lock:
                    decode_progress.pop(session_id, None)
                break
            time.sleep(0.3)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
