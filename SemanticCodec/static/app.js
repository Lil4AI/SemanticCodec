document.addEventListener('DOMContentLoaded', () => {
    // --- Tabs Navigation ---
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.target);
            if (target) target.classList.add('active');
        });
    });

    // --- Utility Functions ---
    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function setLoading(btn, loader, isLoading) {
        btn.disabled = isLoading;
        const btnText = btn.querySelector('.btn-text');
        if (btnText) btnText.style.opacity = isLoading ? '0' : '1';
        if (loader) {
            loader.style.display = isLoading ? 'block' : 'none';
            loader.style.position = isLoading ? 'absolute' : 'static';
        }
    }

    // --- File Dropzone Setup ---
    function setupDropzone(dropzoneId, inputId, onFileSelected) {
        const dropzone = document.getElementById(dropzoneId);
        const input = document.getElementById(inputId);
        const textElem = dropzone.querySelector('.upload-text');

        dropzone.addEventListener('click', () => input.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');

            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                updateFileName(input, textElem, dropzoneId);
                if (onFileSelected) onFileSelected(input.files[0]);
            }
        });

        input.addEventListener('change', () => {
            updateFileName(input, textElem, dropzoneId);
            if (input.files.length && onFileSelected) {
                onFileSelected(input.files[0]);
            }
        });
    }

    function updateFileName(input, textElem, dropzoneId) {
        if (input.files.length > 0) {
            textElem.textContent = input.files[0].name;
            textElem.classList.add('file-selected');
        } else {
            textElem.textContent = dropzoneId === 'encode-dropzone'
                ? 'Drag & Drop an image here or click to browse'
                : 'Drag & Drop a .semic or .semic.jpg file here';
            textElem.classList.remove('file-selected');
        }
    }

    // --- Advanced Settings Toggle ---
    const toggleBtn = document.getElementById('toggle-advanced-btn');
    const advancedPanel = document.getElementById('advanced-panel');
    const toggleArrow = document.getElementById('toggle-arrow');

    if (toggleBtn && advancedPanel) {
        toggleBtn.addEventListener('click', () => {
            const isHidden = advancedPanel.style.display === 'none';
            advancedPanel.style.display = isHidden ? 'block' : 'none';
            if (toggleArrow) {
                toggleArrow.textContent = isHidden ? '▾' : '▸';
            }
        });
    }

    // --- Encode Workflow ---
    setupDropzone('encode-dropzone', 'encode-file');

    const encodeForm = document.getElementById('encode-form');
    const encodeBtn = document.getElementById('encode-btn');
    const encodeLoader = encodeBtn ? encodeBtn.querySelector('.loader') : null;
    const statsContainer = document.getElementById('stats-container');

    if (encodeForm) {
        encodeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('encode-file');

            if (!fileInput.files.length) {
                alert('Please select an image file first.');
                return;
            }

            const mode = document.querySelector('input[name="encode-mode"]:checked').value;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('mode', mode);

            const loraModel = document.getElementById('lora-model').value.trim();
            const loraScale = document.getElementById('lora-scale').value;
            if (loraModel) {
                formData.append('lora_model', loraModel);
                formData.append('lora_scale', loraScale);
            }

            setLoading(encodeBtn, encodeLoader, true);
            if (statsContainer) statsContainer.style.display = 'none';

            try {
                const res = await fetch('/api/encode', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || 'Server error during encoding');
                }

                const data = await res.json();

                if (data.success) {
                    const stats = data.stats;
                    document.getElementById('stat-original').textContent = formatBytes(stats.original_size);
                    document.getElementById('stat-semic').textContent = formatBytes(stats.semic_size);
                    document.getElementById('stat-ratio').textContent = (stats.compression_ratio_pct != null ? stats.compression_ratio_pct : stats.compression_ratio) + '% red.';
                    document.getElementById('stat-bpp').textContent = stats.bpp ? stats.bpp + ' bpp' : '—';
                    document.getElementById('stat-fg-ratio').textContent = stats.fg_area_ratio_pct != null ? stats.fg_area_ratio_pct + '%' : '—';
                    document.getElementById('stat-ratio-display').textContent = stats.ratio_display || '—';

                    // Insights preview
                    const styleBadge = document.getElementById('stat-style-badge');
                    if (styleBadge) styleBadge.textContent = data.image_style || 'photo';
                    const dimsSpan = document.getElementById('stat-dims');
                    if (dimsSpan) dimsSpan.textContent = `${stats.width} × ${stats.height}`;
                    const promptBox = document.getElementById('stat-prompt-preview');
                    if (promptBox) promptBox.textContent = `"${data.prompt || ''}"`;

                    // Component Breakdown Bars
                    const components = stats.components || {};
                    const totalComp = (components.fg_color_webp || 0) + (components.fg_alpha_webp || 0)
                                    + (components.depth_map_webp || 0) + (components.thumb_jpg || 0)
                                    + (components.seg_map_png || components.seg_map_webp || 0);

                    setBars('bar-fg-color', 'size-fg-color', components.fg_color_webp || 0, totalComp);
                    setBars('bar-fg-alpha', 'size-fg-alpha', components.fg_alpha_webp || 0, totalComp);
                    setBars('bar-depth', 'size-depth', components.depth_map_webp || 0, totalComp);
                    setBars('bar-thumb', 'size-thumb', components.thumb_jpg || 0, totalComp);
                    setBars('bar-seg', 'size-seg', components.seg_map_png || components.seg_map_webp || 0, totalComp);

                    // Download button
                    const downloadBtn = document.getElementById('stats-download-btn');
                    downloadBtn.href = `/api/encode/download/${data.download_id}?name=${encodeURIComponent(data.filename)}`;
                    downloadBtn.download = data.filename;

                    statsContainer.style.display = 'block';
                    statsContainer.classList.add('fade-in');
                }
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                setLoading(encodeBtn, encodeLoader, false);
            }
        });
    }

    function setBars(barId, sizeId, bytes, total) {
        const bar = document.getElementById(barId);
        const sizeElem = document.getElementById(sizeId);
        if (!bar || !sizeElem) return;
        const pct = total > 0 ? (bytes / total) * 100 : 0;
        bar.style.width = Math.max(pct, 2) + '%';
        sizeElem.textContent = formatBytes(bytes);
    }

    // --- Decode & Inspect Workflow ---
    let inspectedLitePreview = null;

    async function handleFileInspect(file) {
        if (!file.name.endsWith('.semic') && !file.name.endsWith('.semic.jpg') && !file.name.endsWith('.jpg')) {
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/inspect', {
                method: 'POST',
                body: formData
            });

            if (res.ok) {
                const info = await res.json();
                if (info.success) {
                    const banner = document.getElementById('inspect-banner');
                    if (banner) {
                        banner.style.display = 'block';
                        document.getElementById('inspect-badge').textContent = info.encoded_mode || 'Standard';
                        document.getElementById('inspect-title').textContent = `${info.width} × ${info.height} (${formatBytes(info.total_size)})`;
                        document.getElementById('inspect-bpp').textContent = `${info.bpp} bpp`;
                        document.getElementById('inspect-prompt').textContent = `BLIP: "${info.prompt}"`;
                    }

                    // Auto-fill prompt override textarea with current file's archived prompt
                    const promptArea = document.getElementById('param-prompt-override');
                    if (promptArea) {
                        promptArea.value = info.prompt || '';
                        promptArea.dataset.originalPrompt = info.prompt || '';
                    }

                    // Preload Lite Preview for Before/After Slider
                    if (info.lite_preview) {
                        inspectedLitePreview = info.lite_preview;
                        const beforeImg = document.getElementById('before-image');
                        if (beforeImg) beforeImg.src = inspectedLitePreview;
                    }
                }
            }
        } catch (e) {
            console.warn('Inspect error:', e);
        }
    }

    setupDropzone('decode-dropzone', 'decode-file', handleFileInspect);

    // Auto-sync Guidance Scale when switching decode modes (Turbo = 1.8, Standard/HighFi = 7.5)
    const decodeModeRadios = document.querySelectorAll('input[name="decode-mode"]');
    const guidanceSlider = document.getElementById('param-guidance');
    const guidanceLabel = document.getElementById('val-guidance');
    decodeModeRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (!guidanceSlider) return;
            const defaultGs = radio.value === 'Turbo' ? '1.8' : '7.5';
            guidanceSlider.value = defaultGs;
            delete guidanceSlider.dataset.userModified;
            if (guidanceLabel) guidanceLabel.textContent = defaultGs;
        });
    });

    const decodeForm = document.getElementById('decode-form');
    const decodeBtn = document.getElementById('decode-btn');
    const decodeLoader = decodeBtn ? decodeBtn.querySelector('.loader') : null;
    const resultContainer = document.getElementById('result-container');
    const afterImage = document.getElementById('after-image');
    const beforeImage = document.getElementById('before-image');
    const downloadBtn = document.getElementById('download-btn');
    const progressContainer = document.getElementById('decode-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressLabel = document.getElementById('progress-label');
    const progressEta = document.getElementById('progress-eta');

    if (decodeForm) {
        decodeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('decode-file');

            if (!fileInput.files.length) {
                alert('Please select a .semic or .semic.jpg file first.');
                return;
            }

            const mode = document.querySelector('input[name="decode-mode"]:checked').value;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('mode', mode);

            const sessionId = `dec_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
            formData.append('session_id', sessionId);

            // Advanced params
            formData.append('controlnet_scale', document.getElementById('param-cn-scale').value);
            formData.append('seg_scale', document.getElementById('param-seg-scale').value);
            formData.append('strength', document.getElementById('param-strength').value);
            const guidanceEl = document.getElementById('param-guidance');
            if (guidanceEl && guidanceEl.dataset.userModified === 'true') {
                formData.append('guidance_scale', guidanceEl.value);
            }
            formData.append('dof_strength', document.getElementById('param-dof-strength').value);

            const harmonize = document.getElementById('param-color-harmonize').checked;
            formData.append('color_harmonize', harmonize ? 'true' : 'false');

            const promptEl = document.getElementById('param-prompt-override');
            const customPrompt = promptEl ? promptEl.value.trim() : '';
            const origPrompt = promptEl ? (promptEl.dataset.originalPrompt || '').trim() : '';
            if (customPrompt && customPrompt !== origPrompt) {
                formData.append('custom_prompt', customPrompt);
            }

            const steps = document.getElementById('param-steps').value;
            if (steps && parseInt(steps) > 0) {
                formData.append('steps', steps);
            }

            setLoading(decodeBtn, decodeLoader, true);
            if (resultContainer) resultContainer.style.display = 'none';
            if (progressContainer) {
                progressContainer.style.display = 'block';
                progressFill.style.width = '0%';
                progressLabel.textContent = 'Connecting...';
                progressEta.textContent = 'ETA: --';
            }

            // Connect SSE stream
            let eventSource = new EventSource(`/api/progress/${sessionId}`);
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const pct = data.total > 0 ? (data.step / data.total) * 100 : 0;
                    progressFill.style.width = pct + '%';
                    progressLabel.textContent = `[${data.stage || 'generating'}] Step ${data.step}/${data.total}`;
                    if (data.eta_seconds > 0) {
                        progressEta.textContent = `ETA: ${data.eta_seconds}s`;
                    } else {
                        progressEta.textContent = data.done ? 'Complete!' : 'ETA: --';
                    }
                    if (data.done && eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                } catch (err) {
                    console.error('SSE JSON error', err);
                }
            };
            eventSource.onerror = () => {
                if (eventSource) { eventSource.close(); eventSource = null; }
            };

            try {
                const res = await fetch('/api/decode', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || 'Server decode failed');
                }

                const decodeTime = res.headers.get('X-Decode-Time-Sec') || '—';
                const decodeModeUsed = res.headers.get('X-Decode-Mode') || mode;
                let decodeInfo = null;
                const rawDecodeInfo = res.headers.get('X-Decode-Info');
                if (rawDecodeInfo) {
                    try { decodeInfo = JSON.parse(rawDecodeInfo); } catch (e) {}
                }

                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                afterImage.src = url;
                downloadBtn.href = url;
                downloadBtn.download = `reconstructed_${decodeModeUsed.toLowerCase()}.jpg`;

                // If beforeImage not pre-loaded by inspect, fallback to decoded
                if (!beforeImage.src || beforeImage.src === window.location.href) {
                    beforeImage.src = inspectedLitePreview || url;
                }

                // Update result badges
                const modeTag = document.getElementById('result-mode-tag');
                if (modeTag) modeTag.textContent = decodeModeUsed;
                const timeTag = document.getElementById('result-time-tag');
                if (timeTag) timeTag.textContent = `Decode time: ${decodeTime}s`;

                const metaTag = document.getElementById('result-meta-tag');
                if (metaTag) {
                    const parts = [];
                    if (decodeInfo && decodeInfo.scheduler) {
                        parts.push(`${decodeInfo.scheduler} (${decodeInfo.steps || 0} steps)`);
                    }
                    if (decodeInfo && decodeInfo.metrics) {
                        if (decodeInfo.metrics.fg_psnr_db != null) {
                            parts.push(`FG PSNR: ${decodeInfo.metrics.fg_psnr_db} dB`);
                        }
                        if (decodeInfo.metrics.bg_color_fidelity != null) {
                            parts.push(`Color Fidelity: ${decodeInfo.metrics.bg_color_fidelity}`);
                        }
                    }
                    metaTag.textContent = parts.length ? '• ' + parts.join(' • ') : '';
                }

                const warnBox = document.getElementById('result-warnings');
                if (warnBox) {
                    const warns = (decodeInfo && Array.isArray(decodeInfo.warnings)) ? decodeInfo.warnings : [];
                    if (warns.length > 0) {
                        warnBox.style.display = 'block';
                        warnBox.textContent = 'Note: ' + warns.join(' | ');
                    } else {
                        warnBox.style.display = 'none';
                        warnBox.textContent = '';
                    }
                }

                if (progressContainer) progressContainer.style.display = 'none';
                if (resultContainer) {
                    resultContainer.style.display = 'block';
                    resultContainer.classList.add('fade-in');
                }

                // Reset comparison slider position to 50%
                setSliderPosition(50);

            } catch (err) {
                alert(`Decode error: ${err.message}`);
                if (progressContainer) progressContainer.style.display = 'none';
            } finally {
                if (eventSource) { eventSource.close(); }
                setLoading(decodeBtn, decodeLoader, false);
            }
        });
    }

    // --- Interactive Before / After Comparison Slider ---
    const sliderContainer = document.getElementById('comparison-slider');
    const overlay = document.getElementById('comparison-overlay');
    const handle = document.getElementById('comparison-handle');
    let isSliding = false;

    function setSliderPosition(percentage) {
        percentage = Math.max(0, Math.min(100, percentage));
        if (overlay) overlay.style.clipPath = `inset(0 ${100 - percentage}% 0 0)`;
        if (handle) handle.style.left = percentage + '%';
    }

    if (sliderContainer) {
        function moveSlider(e) {
            if (!isSliding) return;
            const rect = sliderContainer.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const offsetX = clientX - rect.left;
            const pct = (offsetX / rect.width) * 100;
            setSliderPosition(pct);
        }

        sliderContainer.addEventListener('mousedown', (e) => {
            isSliding = true;
            moveSlider(e);
        });
        window.addEventListener('mousemove', moveSlider);
        window.addEventListener('mouseup', () => { isSliding = false; });

        sliderContainer.addEventListener('touchstart', (e) => {
            isSliding = true;
            moveSlider(e);
        });
        window.addEventListener('touchmove', moveSlider);
        window.addEventListener('touchend', () => { isSliding = false; });
    }
});
