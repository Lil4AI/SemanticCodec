document.addEventListener('DOMContentLoaded', () => {
    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // File Upload Handlers
    function setupDropzone(dropzoneId, inputId) {
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
            }
        });

        input.addEventListener('change', () => {
            updateFileName(input, textElem, dropzoneId);
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

    setupDropzone('encode-dropzone', 'encode-file');
    setupDropzone('decode-dropzone', 'decode-file');

    // --- Advanced Settings Toggle ---
    const toggleBtn = document.getElementById('toggle-advanced-btn');
    const advancedPanel = document.getElementById('advanced-panel');
    const toggleArrow = document.getElementById('toggle-arrow');

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const isHidden = advancedPanel.style.display === 'none';
            advancedPanel.style.display = isHidden ? 'block' : 'none';
            toggleArrow.textContent = isHidden ? '▾' : '▸';
            toggleArrow.style.transform = isHidden ? 'rotate(0deg)' : 'rotate(0deg)';
        });
    }

    // --- Utility functions ---
    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // --- Encode Form ---
    const encodeForm = document.getElementById('encode-form');
    const encodeBtn = document.getElementById('encode-btn');
    const encodeLoader = encodeBtn.querySelector('.loader');
    const statsContainer = document.getElementById('stats-container');

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
        statsContainer.style.display = 'none';

        try {
            const res = await fetch('/api/encode', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Server error');
            }

            const data = await res.json();

            if (data.success) {
                // Display compression stats
                const stats = data.stats;
                document.getElementById('stat-original').textContent = formatBytes(stats.original_size);
                document.getElementById('stat-semic').textContent = formatBytes(stats.semic_size);
                document.getElementById('stat-ratio').textContent = stats.compression_ratio + '% reduced';
                document.getElementById('stat-ratio-display').textContent = stats.ratio_display;

                // Component breakdown bars
                const components = stats.components;
                const totalComp = components.fg_color_webp + components.fg_alpha_webp
                                + components.depth_map_webp + components.thumb_jpg;

                setBars('bar-fg-color', 'size-fg-color', components.fg_color_webp, totalComp);
                setBars('bar-fg-alpha', 'size-fg-alpha', components.fg_alpha_webp, totalComp);
                setBars('bar-depth', 'size-depth', components.depth_map_webp, totalComp);
                setBars('bar-thumb', 'size-thumb', components.thumb_jpg, totalComp);

                // Download link
                const downloadBtn = document.getElementById('stats-download-btn');
                downloadBtn.href = `/api/encode/download/${data.download_id}`;
                downloadBtn.download = data.filename;

                statsContainer.style.display = 'block';
                statsContainer.classList.add('fade-in');
            }

        } catch (err) {
            alert(`Error encoding file: ${err.message}`);
        } finally {
            setLoading(encodeBtn, encodeLoader, false);
        }
    });

    function setBars(barId, sizeId, bytes, total) {
        const pct = total > 0 ? (bytes / total) * 100 : 0;
        document.getElementById(barId).style.width = Math.max(pct, 2) + '%';
        document.getElementById(sizeId).textContent = formatBytes(bytes);
    }

    // --- Decode Form ---
    const decodeForm = document.getElementById('decode-form');
    const decodeBtn = document.getElementById('decode-btn');
    const decodeLoader = decodeBtn.querySelector('.loader');
    const resultContainer = document.getElementById('result-container');
    const resultImage = document.getElementById('result-image');
    const downloadBtn = document.getElementById('download-btn');

    decodeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('decode-file');

        if (!fileInput.files.length) {
            alert('Please select a .semic file first.');
            return;
        }

        const mode = document.querySelector('input[name="decode-mode"]:checked').value;
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('mode', mode);

        // Add advanced parameters
        const cnScale = document.getElementById('param-cn-scale').value;
        const strength = document.getElementById('param-strength').value;
        const guidance = document.getElementById('param-guidance').value;
        const steps = document.getElementById('param-steps').value;

        formData.append('controlnet_scale', cnScale);
        formData.append('strength', strength);
        formData.append('guidance_scale', guidance);
        if (steps && parseInt(steps) > 0) {
            formData.append('steps', steps);
        }

        setLoading(decodeBtn, decodeLoader, true);
        resultContainer.style.display = 'none';

        try {
            const res = await fetch('/api/decode', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Server error');
            }

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            resultImage.src = url;
            downloadBtn.href = url;
            downloadBtn.download = `reconstructed_${mode}.jpg`;

            resultContainer.style.display = 'block';

        } catch (err) {
            alert(`Error decoding file: ${err.message}`);
        } finally {
            setLoading(decodeBtn, decodeLoader, false);
        }
    });

    function setLoading(btn, loader, isLoading) {
        btn.disabled = isLoading;
        btn.querySelector('.btn-text').style.opacity = isLoading ? '0' : '1';
        loader.style.display = isLoading ? 'block' : 'none';
        loader.style.position = isLoading ? 'absolute' : 'static';
    }
});
