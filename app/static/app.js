// IsoFlow-S2ST Studio Frontend App Logic

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const videoFileInput = document.getElementById('videoFileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const removeFileBtn = document.getElementById('removeFileBtn');
    
    const languageSelect = document.getElementById('languageSelect');
    const stepsSlider = document.getElementById('stepsSlider');
    const stepsValue = document.getElementById('stepsValue');
    const translateBtn = document.getElementById('translateBtn');

    const processingState = document.getElementById('processingState');
    const statusHeading = document.getElementById('statusHeading');
    const statusDetail = document.getElementById('statusDetail');
    const progressBarFill = document.getElementById('progressBarFill');

    const playerContainer = document.getElementById('playerContainer');
    const placeholderPlayer = document.getElementById('placeholderPlayer');
    const videoPreviewWrapper = document.getElementById('videoPreviewWrapper');
    const outputVideoPlayer = document.getElementById('outputVideoPlayer');

    const metricsBar = document.getElementById('metricsBar');
    const metricAlpha = document.getElementById('metricAlpha');
    const metricFrames = document.getElementById('metricFrames');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');

    let selectedFile = null;

    // Slider listener
    stepsSlider.addEventListener('input', (e) => {
        stepsValue.textContent = `${e.target.value} Steps`;
    });

    // Drag & drop handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        videoFileInput.value = '';
        fileInfo.style.display = 'none';
        document.querySelector('.upload-icon').style.display = 'block';
        document.querySelector('.upload-text').style.display = 'flex';
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        fileInfo.style.display = 'flex';
        document.querySelector('.upload-icon').style.display = 'none';
        document.querySelector('.upload-text').style.display = 'none';
    }

    // Translation Form Submission
    translateBtn.addEventListener('click', async () => {
        const targetLang = languageSelect.value;
        const flowSteps = stepsSlider.value;

        // UI transition to processing state
        translateBtn.disabled = true;
        processingState.style.display = 'flex';
        placeholderPlayer.style.display = 'none';
        videoPreviewWrapper.style.display = 'none';
        metricsBar.style.display = 'none';

        // Simulate progress stages
        updateProgress(15, "Processing Input Video...", "Demuxing audio spectrograms & extracting face frames...");

        const formData = new FormData();
        if (selectedFile) {
            formData.append('file', selectedFile);
        }
        formData.append('target_lang', targetLang);
        formData.append('flow_steps', flowSteps);

        setTimeout(() => updateProgress(45, "Executing Streaming S2ST Encoder...", "Extracting continuous semantic representations..."), 800);
        setTimeout(() => updateProgress(75, "IsoSync-Bridge Duration Regulation...", "Aligning acoustic target pacing & computing Z_shared..."), 1600);
        setTimeout(() => updateProgress(90, "DiFlowDubber ODE Solver...", "Solving Optimal Transport Flow Matching trajectory..."), 2400);

        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            updateProgress(100, "Translation Complete!", "Rendering final output video...");

            setTimeout(() => {
                processingState.style.display = 'none';
                videoPreviewWrapper.style.display = 'block';
                outputVideoPlayer.src = data.video_url;
                outputVideoPlayer.play();

                // Show metrics
                metricAlpha.textContent = `${data.metrics.isochrony_alpha || 1.05}x`;
                metricFrames.textContent = `${data.metrics.num_frames || 50} frames`;
                downloadVideoBtn.href = data.video_url;
                metricsBar.style.display = 'flex';

                translateBtn.disabled = false;
            }, 600);

        } catch (error) {
            console.error('Translation error:', error);
            statusHeading.textContent = "Processing Error";
            statusDetail.textContent = error.message || "Failed to process video.";
            progressBarFill.style.background = "#ef4444";
            translateBtn.disabled = false;
        }
    });

    function updateProgress(percent, heading, detail) {
        progressBarFill.style.width = `${percent}%`;
        statusHeading.textContent = heading;
        statusDetail.textContent = detail;
    }
});
