#!/usr/bin/env python3
"""SemanticCodec v1.0.0 — Flask Web API & SSE server.

Thin HTTP layer over :mod:`codec_engine`. All codec logic lives in the engine.

Endpoints
---------
GET  /                                  Web UI
GET  /api/status                        Engine / model / GPU status
POST /api/encode                        image -> .semic.jpg (JSON stats + download id)
GET  /api/encode/download/<download_id> download the encoded file
POST /api/inspect                       .semic.jpg -> manifest, components, Lite preview
POST /api/decode                        .semic.jpg -> JPEG (Lite / Turbo / Standard / HighFi)
GET  /api/progress/<session_id>         Server-Sent Events decode progress
"""

import io
import json
import os
import random
import threading
import time

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import codec_engine as engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['TEMP_FOLDER'] = os.path.join(BASE_DIR, 'temp_encode')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

TEMP_MAX_AGE_SEC = 60 * 60          # encoded files are kept for 1 hour
SSE_IDLE_TIMEOUT_SEC = 15 * 60      # stop streaming if a session never starts / ends

MODELS = engine.ModelManager()

# --- SSE progress state ---
decode_progress = {}   # session_id -> {step, total, eta_seconds, stage, done, error}
progress_lock = threading.Lock()


def _set_progress(session_id, **kwargs):
    with progress_lock:
        cur = decode_progress.get(session_id, {'step': 0, 'total': 1, 'eta_seconds': 0,
                                               'stage': 'queued', 'done': False})
        cur.update(kwargs)
        cur['updated'] = time.time()
        decode_progress[session_id] = cur


def cleanup_old_temp_files(max_age_sec=TEMP_MAX_AGE_SEC):
    """Delete encoded .semic.jpg files older than ``max_age_sec`` from temp_encode/."""
    now = time.time()
    removed = 0
    folder = app.config['TEMP_FOLDER']
    for name in os.listdir(folder):
        if not name.endswith('.semic.jpg'):
            continue
        path = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(path) > max_age_sec:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def _form_float(name, default=None):
    raw = request.form.get(name, '')
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _form_bool(name, default=True):
    raw = request.form.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'on', 'yes')


@app.route('/')
def index():
    return render_template('index.html', version=engine.VERSION)


@app.route('/api/status')
def status():
    return jsonify(MODELS.status())


@app.route('/api/encode', methods=['POST'])
def encode_image():
    cleanup_old_temp_files()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    mode = request.form.get('mode', 'Standard')
    if mode not in engine.ENCODE_PROFILES:
        return jsonify({'error': f'Unknown mode {mode}'}), 400
    lora_model = os.path.basename(request.form.get('lora_model', '').strip())
    lora_scale = _form_float('lora_scale', 1.0)

    try:
        data = file.read()
        result = engine.encode_to_semic(
            data, mode, models=MODELS, lora_model=lora_model, lora_scale=lora_scale,
            original_file_size=len(data))

        stem = os.path.splitext(secure_filename(file.filename) or 'image')[0] or 'image'
        temp_id = f"encoded_{int(time.time())}_{random.randint(1000, 9999)}"
        with open(os.path.join(app.config['TEMP_FOLDER'], f"{temp_id}.semic.jpg"), 'wb') as f:
            f.write(result.data)

        stats = result.stats
        return jsonify({
            'success': True,
            'download_id': temp_id,
            'filename': f'{stem}.semic.jpg',
            'stats': stats,
            'bpp': stats['bpp'],
            'fg_area_ratio_pct': stats['fg_area_ratio_pct'],
            'prompt': result.manifest['prompt'],
            'negative_prompt': result.manifest['negative_prompt'],
            'image_style': result.manifest['image_style'],
            'manifest': result.manifest,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/encode/download/<download_id>', methods=['GET'])
def download_encoded(download_id):
    safe_id = secure_filename(download_id)
    temp_path = os.path.join(app.config['TEMP_FOLDER'], f"{safe_id}.semic.jpg")
    if not safe_id or not os.path.exists(temp_path):
        return jsonify({'error': 'File not found or expired'}), 404
    name = secure_filename(request.args.get('name', '')) or f'{safe_id}.semic.jpg'
    if not name.endswith('.semic.jpg'):
        name += '.semic.jpg'
    return send_file(temp_path, mimetype='image/jpeg', as_attachment=True, download_name=name)


@app.route('/api/inspect', methods=['POST'])
def inspect_file():
    """Light-weight manifest inspection + instant Lite preview (Before image)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    try:
        payload = engine.load_semic(request.files['file'].read())
        info = engine.inspect_semic(payload)
        info['lite_preview'] = engine.lite_preview_data_url(payload)
        return jsonify({'success': True, **info})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/decode', methods=['POST'])
def decode_image():
    session_id = secure_filename(request.form.get('session_id', '')) or \
        f"dec_{int(time.time())}_{random.randint(1000, 9999)}"
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    mode = request.form.get('mode', 'Standard')
    if mode not in engine.DECODE_PROFILES:
        return jsonify({'error': f'Unknown mode {mode}'}), 400
    steps_raw = _form_float('steps', 0) or 0

    _set_progress(session_id, step=0, total=1, stage='queued', done=False)
    start = time.time()

    def on_progress(stage, step=0, total=1, elapsed=None):
        eta = 0.0
        if stage == 'generating' and step > 0 and elapsed:
            eta = elapsed / step * max(0, total - step)
        _set_progress(session_id, stage=stage, step=int(step), total=max(1, int(total)),
                      eta_seconds=eta, done=False)

    try:
        payload = engine.load_semic(file.read())
        result = engine.decode_semic(
            payload, mode, models=MODELS,
            controlnet_scale=_form_float('controlnet_scale', 0.5),
            seg_scale=_form_float('seg_scale', 0.4),
            strength=_form_float('strength', engine.DEFAULT_STRENGTH),
            guidance_scale=_form_float('guidance_scale', None),
            steps=int(steps_raw) if steps_raw > 0 else None,
            dof_strength=_form_float('dof_strength', 0.0),
            color_harmonize=_form_bool('color_harmonize', True),
            color_lock_strength=_form_float('color_lock_strength', engine.DEFAULT_COLOR_LOCK),
            custom_prompt=request.form.get('custom_prompt', ''),
            progress=on_progress)
        jpeg = result.to_jpeg_bytes()
        info = result.public_info()
        _set_progress(session_id, stage='done', step=max(1, info.get('steps', 1)),
                      total=max(1, info.get('steps', 1)), eta_seconds=0, done=True)

        response = send_file(io.BytesIO(jpeg), mimetype='image/jpeg', as_attachment=True,
                             download_name=f'decoded_{info["mode_used"]}.jpg')
        response.headers['X-Session-Id'] = session_id
        response.headers['X-Decode-Time-Sec'] = f"{time.time() - start:.2f}"
        response.headers['X-Decode-Mode'] = info['mode_used']
        response.headers['X-Decode-Info'] = json.dumps({
            k: info.get(k) for k in ('mode_requested', 'mode_used', 'steps', 'scheduler',
                                     'guidance_scale', 'strength', 'color_harmonize',
                                     'prompt_overridden', 'seed', 'metrics', 'warnings')
        }, ensure_ascii=True)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_progress(session_id, stage='error', done=True, error=str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events endpoint for decode progress."""
    def generate():
        last = None
        opened = time.time()
        while True:
            with progress_lock:
                info = dict(decode_progress.get(session_id, {}))
            snapshot = (info.get('step'), info.get('total'), info.get('stage'), info.get('done'))
            if info and snapshot != last:
                data = {
                    'step': info.get('step', 0), 'total': info.get('total', 1),
                    'eta_seconds': round(info.get('eta_seconds', 0) or 0, 1),
                    'stage': info.get('stage', ''), 'done': info.get('done', False),
                    'error': info.get('error'),
                }
                yield f"data: {json.dumps(data)}\n\n"
                last = snapshot
            if info.get('done'):
                with progress_lock:
                    decode_progress.pop(session_id, None)
                break
            if time.time() - info.get('updated', opened) > SSE_IDLE_TIMEOUT_SEC:
                break
            time.sleep(0.3)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print(f"SemanticCodec v{engine.VERSION} - http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
