/**
 * Indian Traffic AI Counter - Frontend Application Logic
 * Interactive Canvas Line Setting, Real-Time Progress Polling,
 * Dual Video Player + Excel Sheet Preview ("Excel of the Excel")
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Hardware & Client GPU Worker Status Check ---
    function checkHardwareStatus() {
        fetch('/api/hardware-status')
            .then(res => res.json())
            .then(hw => {
                const label = document.getElementById('gpuDeviceLabel');
                const dot = document.getElementById('gpuStatusDot');
                const badge = document.getElementById('gpuBadge');
                const workerLabel = document.getElementById('workerStatusLabel');
                if (!label) return;

                if (hw.connected_workers && hw.connected_workers.length > 0) {
                    const primary = hw.connected_workers[0];
                    label.textContent = `${primary.gpu_name} (Client: ${primary.name})`;
                    label.style.color = '#38bdf8';
                    if (badge) badge.title = `Remote Client GPU Acceleration Active!\nClient: ${primary.name} (${primary.hostname})\nGPU: ${primary.gpu_name}\nCUDA: ${primary.cuda_version}\nVRAM: ${primary.vram_gb}\nTotal Connected Workers: ${hw.connected_workers.length}`;
                    if (dot) {
                        dot.style.background = '#38bdf8';
                        dot.classList.add('pulse');
                    }
                    if (workerLabel) {
                        workerLabel.textContent = `${hw.connected_workers.length} Client GPU Online`;
                        workerLabel.style.color = '#38bdf8';
                    }
                } else if (hw.has_cuda) {
                    label.textContent = `${hw.device_name}`;
                    label.style.color = '#10b981';
                    if (badge) badge.title = `Local Server CUDA Hardware Acceleration Active\nDevice: ${hw.device_name}\nCUDA: ${hw.cuda_version} | PyTorch: ${hw.torch_version}\nFree VRAM: ${hw.vram_free_gb} / ${hw.vram_total_gb} GB`;
                    if (dot) {
                        dot.style.background = '#10b981';
                        dot.classList.add('pulse');
                    }
                    if (workerLabel) {
                        workerLabel.textContent = `Get Client (29 MB)`;
                        workerLabel.style.color = '#a5b4fc';
                    }
                } else {
                    label.textContent = `${hw.device_name}`;
                    label.style.color = '#f59e0b';
                    if (badge) badge.title = `Compute: ${hw.device_name}\nUsing multi-threaded CPU engine.\nConnect a Client GPU Worker to accelerate!`;
                    if (dot) {
                        dot.style.background = '#f59e0b';
                        dot.classList.remove('pulse');
                    }
                    if (workerLabel) {
                        workerLabel.textContent = `Get Client (29 MB)`;
                        workerLabel.style.color = '#a5b4fc';
                    }
                }
            })
            .catch(err => console.warn('Could not query hardware status:', err));
    }
    checkHardwareStatus();
    setInterval(checkHardwareStatus, 8000);
    // --- System Version & Auto-Update Logic ---
    let latestUpdateInfo = null;

    function checkSystemUpdates() {
        const verLabel = document.getElementById('appVersionLabel');
        const alertDot = document.getElementById('updateAlertDot');
        
        fetch('/api/update/status')
            .then(res => res.json())
            .then(data => {
                if (verLabel) verLabel.textContent = `v${data.current_version}`;
                return fetch('/api/update/check');
            })
            .then(res => res.json())
            .then(info => {
                latestUpdateInfo = info;
                if (info && info.update_available) {
                    if (alertDot) alertDot.style.display = 'inline-block';
                    if (verLabel) {
                        verLabel.textContent = `v${info.current_version} (Update Available!)`;
                        verLabel.style.color = '#38bdf8';
                    }
                } else {
                    if (alertDot) alertDot.style.display = 'none';
                }
            })
            .catch(err => console.warn('Update check failed:', err));
    }
    checkSystemUpdates();

    // Update Modal handling
    const versionBadge = document.getElementById('versionBadge');
    const updateModal = document.getElementById('updateModal');
    const closeUpdateModal = document.getElementById('closeUpdateModal');
    const cancelUpdateBtn = document.getElementById('cancelUpdateBtn');
    const startUpdateBtn = document.getElementById('startUpdateBtn');
    const modalCurrentVer = document.getElementById('modalCurrentVer');
    const modalRemoteVer = document.getElementById('modalRemoteVer');
    const modalChangelog = document.getElementById('modalChangelog');
    const updateProgressContainer = document.getElementById('updateProgressContainer');
    const updateProgressBar = document.getElementById('updateProgressBar');
    const updateProgressPctText = document.getElementById('updateProgressPctText');
    const updateProgressStatusText = document.getElementById('updateProgressStatusText');
    const startUpdateBtnText = document.getElementById('startUpdateBtnText');

    function openUpdateModal() {
        if (!updateModal) return;
        updateModal.style.display = 'flex';
        
        if (latestUpdateInfo) {
            modalCurrentVer.textContent = `v${latestUpdateInfo.current_version}`;
            modalRemoteVer.textContent = `v${latestUpdateInfo.remote_version}`;
            modalChangelog.textContent = latestUpdateInfo.changelog || 'Performance improvements and bug fixes.';
            if (!latestUpdateInfo.update_available) {
                modalChangelog.textContent = 'Your system is currently on the latest version (v' + latestUpdateInfo.current_version + ').';
                startUpdateBtn.style.display = 'none';
            } else {
                startUpdateBtn.style.display = 'inline-flex';
            }
        }
    }

    if (versionBadge) versionBadge.addEventListener('click', openUpdateModal);
    if (closeUpdateModal) closeUpdateModal.addEventListener('click', () => updateModal.style.display = 'none');
    if (cancelUpdateBtn) cancelUpdateBtn.addEventListener('click', () => updateModal.style.display = 'none');

    if (startUpdateBtn) {
        startUpdateBtn.addEventListener('click', () => {
            if (!latestUpdateInfo || !latestUpdateInfo.download_url) {
                alert('No download package URL found in update manifest.');
                return;
            }
            startUpdateBtn.disabled = true;
            if (cancelUpdateBtn) cancelUpdateBtn.disabled = true;
            updateProgressContainer.style.display = 'block';
            startUpdateBtnText.textContent = 'Updating...';

            fetch('/api/update/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ download_url: latestUpdateInfo.download_url })
            })
            .then(res => res.json())
            .then(data => {
                const pollInterval = setInterval(() => {
                    fetch('/api/update/progress')
                        .then(r => r.json())
                        .then(p => {
                            if (p.status === 'downloading') {
                                updateProgressStatusText.textContent = 'Downloading update package...';
                                updateProgressBar.style.width = `${p.percent}%`;
                                updateProgressPctText.textContent = `${p.percent}%`;
                            } else if (p.status === 'extracting') {
                                updateProgressStatusText.textContent = 'Extracting and staging files...';
                                updateProgressBar.style.width = '100%';
                                updateProgressPctText.textContent = '100%';
                            } else if (p.status === 'restarting') {
                                updateProgressStatusText.textContent = 'Restarting application... please wait';
                                clearInterval(pollInterval);
                                setTimeout(pollServerForReload, 3500);
                            } else if (p.status === 'error') {
                                clearInterval(pollInterval);
                                alert('Update error: ' + p.error);
                                startUpdateBtn.disabled = false;
                                startUpdateBtnText.textContent = 'Retry Update';
                            }
                        })
                        .catch(() => {
                            updateProgressStatusText.textContent = 'System is restarting with new version...';
                            setTimeout(pollServerForReload, 3000);
                        });
                }, 800);
            })
            .catch(err => {
                alert('Failed to initiate update: ' + err);
                startUpdateBtn.disabled = false;
            });
        });
    }

    function pollServerForReload() {
        const pingInterval = setInterval(() => {
            fetch('/api/update/status')
                .then(r => {
                    if (r.ok) {
                        clearInterval(pingInterval);
                        window.location.reload();
                    }
                })
                .catch(() => {
                    // Still restarting, retry in 1.2s
                });
        }, 1200);
    }

    // --- DOM Elements ---
    const dropzone = document.getElementById('dropzone');
    const videoFileInput = document.getElementById('videoFileInput');
    const browseBtn = document.getElementById('browseBtn');
    const loadSampleBtn = document.getElementById('loadSampleBtn');
    const fileInfoBar = document.getElementById('fileInfoBar');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileMetaDisplay = document.getElementById('fileMetaDisplay');
    const removeFileBtn = document.getElementById('removeFileBtn');

    const confSlider = document.getElementById('confSlider');
    const confVal = document.getElementById('confVal');
    const lineYSlider = document.getElementById('lineYSlider');
    const lineYVal = document.getElementById('lineYVal');

    const lineCanvas = document.getElementById('lineCanvas');
    const canvasPlaceholder = document.getElementById('canvasPlaceholder');
    const lineCoordsText = document.getElementById('lineCoordsText');
    const resetLineBtn = document.getElementById('resetLineBtn');
    const startProcessBtn = document.getElementById('startProcessBtn');
    const startBtnText = document.getElementById('startBtnText');

    const setupSection = document.getElementById('setupSection');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');

    // Progress Elements
    const progressBarFill = document.getElementById('progressBarFill');
    const progressPercentText = document.getElementById('progressPercentText');
    const progressFramesText = document.getElementById('progressFramesText');
    const progressEtaText = document.getElementById('progressEtaText');
    const liveFpsText = document.getElementById('liveFpsText');

    const liveTotalCount = document.getElementById('liveTotalCount');
    const liveCarCount = document.getElementById('liveCarCount');
    const liveAutoCount = document.getElementById('liveAutoCount');
    const liveBikeCount = document.getElementById('liveBikeCount');
    const liveBusCount = document.getElementById('liveBusCount');
    const liveTruckCount = document.getElementById('liveTruckCount');
    const liveTempoCount = document.getElementById('liveTempoCount');

    // Results Elements
    const resultsSummaryText = document.getElementById('resultsSummaryText');
    const downloadExcelBtn = document.getElementById('downloadExcelBtn');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');

    const kpiTotal = document.getElementById('kpiTotal');
    const kpiInward = document.getElementById('kpiInward');
    const kpiOutward = document.getElementById('kpiOutward');
    const kpiAuto = document.getElementById('kpiAuto');
    const kpiBike = document.getElementById('kpiBike');
    const kpiCar = document.getElementById('kpiCar');
    const kpiBus = document.getElementById('kpiBus');
    const kpiTruck = document.getElementById('kpiTruck');

    const resultVideoPlayer = document.getElementById('resultVideoPlayer');
    const tableSearchInput = document.getElementById('tableSearchInput');
    const classFilterSelect = document.getElementById('classFilterSelect');
    const dirFilterSelect = document.getElementById('dirFilterSelect');
    const excelTableBody = document.getElementById('excelTableBody');
    const tableCountInfo = document.getElementById('tableCountInfo');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const currentPageText = document.getElementById('currentPageText');
    const breakdownGrid = document.getElementById('breakdownGrid');
    const processAnotherBtn = document.getElementById('processAnotherBtn');

    // --- State Variables ---
    let activeVideoSource = null;      // 'sample' or uploaded filename/path
    let videoResolution = { w: 1280, h: 720 };
    let previewImage = new Image();
    let hasPreviewFrame = false;

    // Line Coordinates in Canvas space (0..640, 0..360)
    const ctx = lineCanvas.getContext('2d');
    let lineStart = { x: 80, y: 216 };
    let lineEnd = { x: 560, y: 216 };
    let activeHandle = null;           // 'start' | 'end' | null
    const HANDLE_RADIUS = 10;

    let currentJobId = null;
    let pollInterval = null;
    let allEventsData = [];
    let filteredEventsData = [];
    let currentPage = 1;
    const ROWS_PER_PAGE = 10;

    // --- INITIALIZATION ---
    initCanvas();
    updateCoordsDisplay();

    // -------------------------------------------------------------
    // CANVAS INTERACTIVE LINE EDITOR
    // -------------------------------------------------------------
    function initCanvas() {
        lineCanvas.width = 640;
        lineCanvas.height = 360;
        drawCanvas();

        lineCanvas.addEventListener('mousedown', onCanvasMouseDown);
        window.addEventListener('mousemove', onCanvasMouseMove);
        window.addEventListener('mouseup', onCanvasMouseUp);

        // Touch support for mobile/tablets
        lineCanvas.addEventListener('touchstart', (e) => {
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            lineCanvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        window.addEventListener('touchmove', (e) => {
            if (!activeHandle) return;
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            window.dispatchEvent(mouseEvent);
        }, { passive: false });

        window.addEventListener('touchend', () => {
            window.dispatchEvent(new MouseEvent('mouseup'));
        });
    }

    let lineDragOffset = null;

    function distToSegment(p, v, w) {
        const l2 = (v.x - w.x) ** 2 + (v.y - w.y) ** 2;
        if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
        let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(p.x - (v.x + t * (w.x - v.x)), p.y - (v.y + t * (w.y - v.y)));
    }

    function getCanvasPos(evt) {
        const rect = lineCanvas.getBoundingClientRect();
        const scaleX = lineCanvas.width / (rect.width || 1);
        const scaleY = lineCanvas.height / (rect.height || 1);
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }

    function onCanvasMouseDown(evt) {
        const pos = getCanvasPos(evt);
        const distStart = Math.hypot(pos.x - lineStart.x, pos.y - lineStart.y);
        const distEnd = Math.hypot(pos.x - lineEnd.x, pos.y - lineEnd.y);
        const hitRadius = Math.max(22, HANDLE_RADIUS * 2.2);

        if (distStart <= hitRadius) {
            activeHandle = 'start';
            lineDragOffset = null;
        } else if (distEnd <= hitRadius) {
            activeHandle = 'end';
            lineDragOffset = null;
        } else if (distToSegment(pos, lineStart, lineEnd) <= 16) {
            activeHandle = 'line';
            lineDragOffset = {
                clickX: pos.x,
                clickY: pos.y,
                sX: lineStart.x,
                sY: lineStart.y,
                eX: lineEnd.x,
                eY: lineEnd.y
            };
        } else {
            activeHandle = null;
            lineDragOffset = null;
        }
    }

    function onCanvasMouseMove(evt) {
        if (!activeHandle) {
            // Hover cursor indication
            const pos = getCanvasPos(evt);
            const distStart = Math.hypot(pos.x - lineStart.x, pos.y - lineStart.y);
            const distEnd = Math.hypot(pos.x - lineEnd.x, pos.y - lineEnd.y);
            const hitRadius = Math.max(22, HANDLE_RADIUS * 2.2);
            if (distStart <= hitRadius || distEnd <= hitRadius) {
                lineCanvas.style.cursor = 'grab';
            } else if (distToSegment(pos, lineStart, lineEnd) <= 16) {
                lineCanvas.style.cursor = 'move';
            } else {
                lineCanvas.style.cursor = 'crosshair';
            }
            return;
        }

        const pos = getCanvasPos(evt);

        if (activeHandle === 'line' && lineDragOffset) {
            lineCanvas.style.cursor = 'grabbing';
            const dx = pos.x - lineDragOffset.clickX;
            const dy = pos.y - lineDragOffset.clickY;
            const minX = Math.min(lineDragOffset.sX, lineDragOffset.eX);
            const maxX = Math.max(lineDragOffset.sX, lineDragOffset.eX);
            const minY = Math.min(lineDragOffset.sY, lineDragOffset.eY);
            const maxY = Math.max(lineDragOffset.sY, lineDragOffset.eY);

            const clampDx = Math.max(10 - minX, Math.min(lineCanvas.width - 10 - maxX, dx));
            const clampDy = Math.max(10 - minY, Math.min(lineCanvas.height - 10 - maxY, dy));

            lineStart.x = lineDragOffset.sX + clampDx;
            lineStart.y = lineDragOffset.sY + clampDy;
            lineEnd.x = lineDragOffset.eX + clampDx;
            lineEnd.y = lineDragOffset.eY + clampDy;
        } else {
            lineCanvas.style.cursor = 'grabbing';
            const clampedX = Math.max(10, Math.min(lineCanvas.width - 10, pos.x));
            const clampedY = Math.max(10, Math.min(lineCanvas.height - 10, pos.y));

            if (activeHandle === 'start') {
                lineStart.x = clampedX;
                lineStart.y = clampedY;
            } else if (activeHandle === 'end') {
                lineEnd.x = clampedX;
                lineEnd.y = clampedY;
            }
        }

        drawCanvas();
        updateCoordsDisplay();
    }

    function onCanvasMouseUp() {
        if (activeHandle) {
            activeHandle = null;
            lineDragOffset = null;
            lineCanvas.style.cursor = 'crosshair';
        }
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);

        // Draw preview image frame if loaded
        if (hasPreviewFrame && previewImage.complete) {
            ctx.drawImage(previewImage, 0, 0, lineCanvas.width, lineCanvas.height);
            // Slight dark tint for line contrast
            ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
            ctx.fillRect(0, 0, lineCanvas.width, lineCanvas.height);
        } else {
            // Dark grid background
            ctx.fillStyle = '#060A14';
            ctx.fillRect(0, 0, lineCanvas.width, lineCanvas.height);

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let x = 0; x < lineCanvas.width; x += 40) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, lineCanvas.height); ctx.stroke();
            }
            for (let y = 0; y < lineCanvas.height; y += 40) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(lineCanvas.width, y); ctx.stroke();
            }
        }

        // Draw Glowing Virtual Line
        ctx.save();
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(lineStart.x, lineStart.y);
        ctx.lineTo(lineEnd.x, lineEnd.y);
        ctx.stroke();

        ctx.strokeStyle = '#06B6D4';
        ctx.lineWidth = 4;
        ctx.stroke();

        // Inner dashed bright core
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 6]);
        ctx.stroke();
        ctx.restore();

        // Direction Indicator (Perpendicular Arrow)
        const midX = (lineStart.x + lineEnd.x) / 2;
        const midY = (lineStart.y + lineEnd.y) / 2;
        const dx = lineEnd.x - lineStart.x;
        const dy = lineEnd.y - lineStart.y;
        const len = Math.hypot(dx, dy) || 1;
        const perpX = -dy / len;
        const perpY = dx / len;
        const arrowLen = 30;

        ctx.save();
        ctx.strokeStyle = '#FACC15';
        ctx.fillStyle = '#FACC15';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(midX + perpX * arrowLen, midY + perpY * arrowLen);
        ctx.stroke();

        // Arrow head
        const tipX = midX + perpX * arrowLen;
        const tipY = midY + perpY * arrowLen;
        const angle = Math.atan2(perpY, perpX);
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - 10 * Math.cos(angle - Math.PI / 6), tipY - 10 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(tipX - 10 * Math.cos(angle + Math.PI / 6), tipY - 10 * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();

        // Flow Label
        ctx.font = '600 11px Inter, sans-serif';
        ctx.fillStyle = '#FACC15';
        ctx.fillText('INWARD FLOW ⬇', midX + perpX * (arrowLen + 15) - 45, midY + perpY * (arrowLen + 15) + 4);
        ctx.restore();

        // Draw Handle A (Start)
        drawHandle(lineStart.x, lineStart.y, 'A (Start)');
        // Draw Handle B (End)
        drawHandle(lineEnd.x, lineEnd.y, 'B (End)');
    }

    function drawHandle(x, y, label) {
        ctx.save();
        // Glow
        ctx.shadowColor = '#FACC15';
        ctx.shadowBlur = 12;

        // Outer Ring
        ctx.fillStyle = '#FACC15';
        ctx.beginPath();
        ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fill();

        // Inner Core
        ctx.fillStyle = '#0F172A';
        ctx.beginPath();
        ctx.arc(x, y, HANDLE_RADIUS - 3, 0, Math.PI * 2);
        ctx.fill();

        // Label Tag
        ctx.shadowBlur = 0;
        ctx.font = '600 10px JetBrains Mono, monospace';
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x - 18, y - 14);
        ctx.restore();
    }

    function updateCoordsDisplay() {
        const scaleX = videoResolution.w / lineCanvas.width;
        const scaleY = videoResolution.h / lineCanvas.height;

        const realX1 = Math.round(lineStart.x * scaleX);
        const realY1 = Math.round(lineStart.y * scaleY);
        const realX2 = Math.round(lineEnd.x * scaleX);
        const realY2 = Math.round(lineEnd.y * scaleY);

        lineCoordsText.textContent = `A (${realX1}, ${realY1}) ➔ B (${realX2}, ${realY2})`;
    }

    function getScaledRealCoords() {
        const scaleX = videoResolution.w / lineCanvas.width;
        const scaleY = videoResolution.h / lineCanvas.height;
        return {
            x1: Math.round(lineStart.x * scaleX),
            y1: Math.round(lineStart.y * scaleY),
            x2: Math.round(lineEnd.x * scaleX),
            y2: Math.round(lineEnd.y * scaleY)
        };
    }

    resetLineBtn.addEventListener('click', () => {
        const defaultY = Math.round(lineCanvas.height * 0.60);
        lineStart = { x: Math.round(lineCanvas.width * 0.12), y: defaultY };
        lineEnd = { x: Math.round(lineCanvas.width * 0.88), y: defaultY };
        lineYSlider.value = 60;
        lineYVal.textContent = '60%';
        drawCanvas();
        updateCoordsDisplay();
    });

    lineYSlider.addEventListener('input', (e) => {
        const pct = parseInt(e.target.value, 10);
        lineYVal.textContent = `${pct}%`;
        const newY = Math.round(lineCanvas.height * (pct / 100));
        lineStart.y = newY;
        lineEnd.y = newY;
        drawCanvas();
        updateCoordsDisplay();
    });

    confSlider.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // -------------------------------------------------------------
    // VIDEO UPLOAD & SAMPLE LOADING
    // -------------------------------------------------------------
    browseBtn.addEventListener('click', () => videoFileInput.click());
    dropzone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) videoFileInput.click();
    });

    ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    loadSampleBtn.addEventListener('click', loadSampleVideo);

    function loadSampleVideo() {
        startBtnText.textContent = 'Loading Sample Video...';
        loadSampleBtn.disabled = true;

        fetch('/api/sample-video', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    activeVideoSource = data.video_path;
                    videoResolution = { w: data.width || 1280, h: data.height || 720 };
                    displayActiveFile(data.filename, `${videoResolution.w}x${videoResolution.h} • ${data.fps} FPS`);
                    loadPreviewFrame(data.frame_url);
                } else {
                    alert('Error loading sample video: ' + (data.message || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error(err);
                alert('Failed to connect to backend server.');
            })
            .finally(() => {
                loadSampleBtn.disabled = false;
            });
    }

    function handleFileUpload(file) {
        startBtnText.textContent = 'Uploading Video...';
        startProcessBtn.disabled = true;

        const formData = new FormData();
        formData.append('video', file);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    activeVideoSource = data.video_path;
                    videoResolution = { w: data.width || 1280, h: data.height || 720 };
                    displayActiveFile(data.filename, `${videoResolution.w}x${videoResolution.h} • ${(file.size / (1024 * 1024)).toFixed(1)} MB`);
                    loadPreviewFrame(data.frame_url);
                } else {
                    alert('Upload failed: ' + (data.message || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error(err);
                alert('Upload failed due to connection error.');
            });
    }

    function displayActiveFile(name, meta) {
        fileNameDisplay.textContent = name;
        fileMetaDisplay.textContent = meta;
        fileInfoBar.classList.remove('hidden');
        dropzone.classList.add('hidden');
        startProcessBtn.disabled = false;
        startBtnText.textContent = 'Start Real-Time Processing on RTX 4070 SUPER';
    }

    removeFileBtn.addEventListener('click', () => {
        activeVideoSource = null;
        hasPreviewFrame = false;
        fileInfoBar.classList.add('hidden');
        dropzone.classList.remove('hidden');
        canvasPlaceholder.classList.remove('hidden');
        startProcessBtn.disabled = true;

        // Reset canvas to default 16:9
        lineCanvas.width = 640;
        lineCanvas.height = 360;
        const canvasWrapper = document.getElementById('canvasWrapper');
        if (canvasWrapper) {
            canvasWrapper.style.aspectRatio = '';
            canvasWrapper.style.maxWidth = '100%';
            canvasWrapper.style.width = 'fit-content';
        }
        lineStart = { x: 80, y: 216 };
        lineEnd = { x: 560, y: 216 };
        drawCanvas();
        updateCoordsDisplay();
    });

    function loadPreviewFrame(url) {
        previewImage = new Image();
        previewImage.crossOrigin = 'anonymous';
        previewImage.onload = () => {
            hasPreviewFrame = true;
            canvasPlaceholder.classList.add('hidden');

            // 1. Get true image dimensions and exact aspect ratio
            const imgW = previewImage.naturalWidth || videoResolution.w || 1280;
            const imgH = previewImage.naturalHeight || videoResolution.h || 720;
            videoResolution.w = imgW;
            videoResolution.h = imgH;
            const aspect = imgW / imgH;

            // 2. Adjust internal canvas resolution to match true video aspect ratio
            if (imgW >= imgH) {
                // Landscape (16:9, 4:3, etc.)
                lineCanvas.width = 640;
                lineCanvas.height = Math.round(640 / aspect);
            } else {
                // Portrait / Vertical (9:16, 3:4, etc.)
                lineCanvas.height = 500;
                lineCanvas.width = Math.round(500 * aspect);
            }

            // 3. Keep wrapper fitted tightly to the canvas
            const canvasWrapper = document.getElementById('canvasWrapper');
            if (canvasWrapper) {
                canvasWrapper.style.aspectRatio = '';
                canvasWrapper.style.maxWidth = '100%';
                canvasWrapper.style.width = 'fit-content';
                canvasWrapper.style.margin = '0 auto';
            }

            // 4. Position endpoints proportionally
            const defaultY = Math.round(lineCanvas.height * (parseInt(lineYSlider.value, 10) / 100));
            lineStart = { x: Math.round(lineCanvas.width * 0.12), y: defaultY };
            lineEnd = { x: Math.round(lineCanvas.width * 0.88), y: defaultY };

            drawCanvas();
            updateCoordsDisplay();
        };
        previewImage.src = url + '?t=' + Date.now();
    }

    // -------------------------------------------------------------
    // START REAL-TIME PROCESSING
    // -------------------------------------------------------------
    startProcessBtn.addEventListener('click', startProcessing);

    function startProcessing() {
        if (!activeVideoSource) {
            alert('Please select or upload a video first.');
            return;
        }

        const coords = getScaledRealCoords();
        const conf = parseFloat(confSlider.value);

        setupSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        // Reset progress UI
        progressBarFill.style.width = '0%';
        progressPercentText.textContent = '0% Complete';
        progressFramesText.textContent = 'Starting GPU worker...';
        progressEtaText.textContent = 'Warming up YOLOv11 CUDA engine...';
        liveFpsText.textContent = '-- FPS';

        fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_source: activeVideoSource,
                line: `${coords.x1},${coords.y1},${coords.x2},${coords.y2}`,
                conf: conf
            })
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    currentJobId = data.job_id;
                    pollInterval = setInterval(pollProgress, 400);
                } else {
                    alert('Failed to start processing: ' + data.message);
                    resetToSetup();
                }
            })
            .catch(err => {
                console.error(err);
                alert('Connection error starting processing job.');
                resetToSetup();
            });
    }

    function pollProgress() {
        if (!currentJobId) return;

        fetch(`/api/progress/${currentJobId}`)
            .then(res => res.json())
            .then(job => {
                if (job.status === 'processing') {
                    const pct = job.percent || 0;
                    progressBarFill.style.width = `${pct}%`;
                    progressPercentText.textContent = `${pct}% Complete`;
                    progressFramesText.textContent = `Frame ${job.current_frame} / ${job.total_frames}`;
                    progressEtaText.textContent = `Elapsed: ${job.elapsed_sec.toFixed(1)}s • ETA: ${job.eta_sec.toFixed(1)}s`;
                    liveFpsText.textContent = `${job.fps.toFixed(1)} FPS`;

                    // Update live ticker counts
                    const counts = job.active_counts || {};
                    liveTotalCount.textContent = job.total_count || 0;
                    liveCarCount.textContent = counts['Car'] || 0;
                    liveAutoCount.textContent = counts['Auto-Rickshaw'] || 0;
                    liveBikeCount.textContent = counts['Bike / Two-Wheeler'] || 0;
                    liveBusCount.textContent = counts['Bus'] || 0;
                    liveTruckCount.textContent = counts['Truck'] || 0;
                    liveTempoCount.textContent = counts['Tempo / LCV'] || 0;
                } else if (job.status === 'completed') {
                    clearInterval(pollInterval);
                    progressBarFill.style.width = '100%';
                    progressPercentText.textContent = '100% Complete';
                    fetchResults(currentJobId);
                } else if (job.status === 'error') {
                    clearInterval(pollInterval);
                    alert('Processing error: ' + job.error);
                    resetToSetup();
                }
            })
            .catch(err => console.error('Progress poll error:', err));
    }

    function resetToSetup() {
        progressSection.classList.add('hidden');
        setupSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    }

    // -------------------------------------------------------------
    // DISPLAY RESULTS, VIDEO PLAYER & EXCEL TABLE PREVIEW
    // -------------------------------------------------------------
    function fetchResults(jobId) {
        fetch(`/api/results/${jobId}`)
            .then(res => res.json())
            .then(data => {
                progressSection.classList.add('hidden');
                resultsSection.classList.remove('hidden');

                const summary = data.summary;
                allEventsData = data.events || [];

                // 1. Results Header Summary
                const deviceName = summary.device_name || 'GPU';
                resultsSummaryText.textContent = `Processed ${summary.total_frames} frames • ${summary.total_count} vehicles counted • ${summary.fps.toFixed(1)} FPS on ${deviceName}`;

                // 2. Download Links
                downloadExcelBtn.href = `/api/download/excel/${jobId}`;
                downloadCsvBtn.href = `/api/download/csv/${jobId}`;
                downloadVideoBtn.href = `/api/video/${jobId}`;

                // 3. KPI Cards
                kpiTotal.textContent = summary.total_count;
                kpiInward.textContent = summary.inward_count;
                kpiOutward.textContent = summary.outward_count;

                const cls = summary.class_counts || {};
                kpiAuto.textContent = cls['Auto-Rickshaw'] || 0;
                kpiBike.textContent = cls['Bike / Two-Wheeler'] || 0;
                kpiCar.textContent = cls['Car'] || 0;
                kpiBus.textContent = cls['Bus'] || 0;
                kpiTruck.textContent = cls['Truck'] || 0;

                // 4. Video Player Preview
                resultVideoPlayer.src = `/api/video/${jobId}?t=${Date.now()}`;
                resultVideoPlayer.load();

                // 5. Excel Table Preview ("Excel of the Excel")
                applyTableFilters();

                // 6. Breakdown Grid
                renderBreakdownGrid(summary);

                // Smooth scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(err => {
                console.error('Error fetching results:', err);
                alert('Failed to load job results.');
            });
    }

    // -------------------------------------------------------------
    // EXCEL DATA TABLE PREVIEW ("EXCEL OF THE EXCEL")
    // -------------------------------------------------------------
    tableSearchInput.addEventListener('input', () => {
        currentPage = 1;
        applyTableFilters();
    });

    classFilterSelect.addEventListener('change', () => {
        currentPage = 1;
        applyTableFilters();
    });

    dirFilterSelect.addEventListener('change', () => {
        currentPage = 1;
        applyTableFilters();
    });

    function applyTableFilters() {
        const query = tableSearchInput.value.trim().toLowerCase();
        const selectedClass = classFilterSelect.value;
        const selectedDir = dirFilterSelect.value;

        filteredEventsData = allEventsData.filter(ev => {
            // Class match
            if (selectedClass !== 'ALL' && ev.class !== selectedClass) return false;
            // Direction match
            if (selectedDir !== 'ALL' && ev.direction !== selectedDir) return false;
            // Text search
            if (query) {
                const combined = `${ev.track_id} ${ev.class} ${ev.raw_class} ${ev.direction} ${ev.point}`.toLowerCase();
                if (!combined.includes(query)) return false;
            }
            return true;
        });

        renderExcelTable();
    }

    function renderExcelTable() {
        excelTableBody.innerHTML = '';
        const total = filteredEventsData.length;

        if (total === 0) {
            excelTableBody.innerHTML = `
                <tr>
                    <td colspan="9" style="text-align: center; padding: 24px; color: var(--text-muted);">
                        No vehicle crossing events match the selected criteria.
                    </td>
                </tr>
            `;
            tableCountInfo.textContent = 'Showing 0 of 0 entries';
            prevPageBtn.disabled = true;
            nextPageBtn.disabled = true;
            return;
        }

        const totalPages = Math.ceil(total / ROWS_PER_PAGE);
        if (currentPage > totalPages) currentPage = totalPages;
        if (currentPage < 1) currentPage = 1;

        const startIdx = (currentPage - 1) * ROWS_PER_PAGE;
        const endIdx = Math.min(startIdx + ROWS_PER_PAGE, total);
        const pageItems = filteredEventsData.slice(startIdx, endIdx);

        pageItems.forEach((ev, i) => {
            const tr = document.createElement('tr');
            const sno = startIdx + i + 1;

            let badgeClass = 'badge-car';
            if (ev.class === 'Auto-Rickshaw') badgeClass = 'badge-auto';
            else if (ev.class === 'Bike / Two-Wheeler') badgeClass = 'badge-bike';
            else if (ev.class === 'Bus') badgeClass = 'badge-bus';
            else if (ev.class === 'Truck') badgeClass = 'badge-truck';
            else if (ev.class === 'Tempo / LCV') badgeClass = 'badge-tempo';

            const dirBadge = ev.direction === 'Inward' 
                ? '<span class="badge-inward">⬇️ Inward</span>' 
                : '<span class="badge-outward">⬆️ Outward</span>';

            const confVal = parseFloat(ev.confidence);
            const confStr = confVal <= 1.0 ? `${(confVal * 100).toFixed(1)}%` : `${confVal.toFixed(1)}%`;

            tr.innerHTML = `
                <td>${sno}</td>
                <td><strong>#${ev.frame}</strong></td>
                <td>${ev.timestamp}</td>
                <td><code style="color: #67E8F9;">#${ev.track_id}</code></td>
                <td><span class="badge-cat ${badgeClass}">${ev.class}</span></td>
                <td style="color: var(--text-muted);">${ev.raw_class}</td>
                <td>${confStr}</td>
                <td>${dirBadge}</td>
                <td style="font-family: var(--font-mono); font-size: 0.75rem;">${ev.point}</td>
            `;
            excelTableBody.appendChild(tr);
        });

        tableCountInfo.textContent = `Showing ${startIdx + 1} to ${endIdx} of ${total} entries`;
        currentPageText.textContent = currentPage;
        prevPageBtn.disabled = currentPage <= 1;
        nextPageBtn.disabled = currentPage >= totalPages;
    }

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderExcelTable();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredEventsData.length / ROWS_PER_PAGE);
        if (currentPage < totalPages) {
            currentPage++;
            renderExcelTable();
        }
    });

    // -------------------------------------------------------------
    // CLASS DISTRIBUTION BREAKDOWN BARS
    // -------------------------------------------------------------
    function renderBreakdownGrid(summary) {
        breakdownGrid.innerHTML = '';
        const counts = summary.class_counts || {};
        const total = summary.total_count || 1;

        const categories = [
            { name: 'Auto-Rickshaw', icon: '🛺', key: 'Auto-Rickshaw' },
            { name: 'Two-Wheeler', icon: '🏍️', key: 'Bike / Two-Wheeler' },
            { name: 'Car', icon: '🚗', key: 'Car' },
            { name: 'Bus', icon: '🚌', key: 'Bus' },
            { name: 'Truck', icon: '🚛', key: 'Truck' },
            { name: 'Tempo / LCV', icon: '🚐', key: 'Tempo / LCV' }
        ];

        categories.forEach(cat => {
            const cnt = counts[cat.key] || 0;
            const pct = ((cnt / total) * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = 'breakdown-item';
            item.innerHTML = `
                <div class="breakdown-top">
                    <span>${cat.icon} ${cat.name}</span>
                    <span><strong>${cnt}</strong> (${pct}%)</span>
                </div>
                <div class="breakdown-bar-bg">
                    <div class="breakdown-bar-fill" style="width: ${pct}%;"></div>
                </div>
                <div class="breakdown-sub">
                    <span>Inward: ${summary.inward_class_counts?.[cat.key] || 0}</span>
                    <span>Outward: ${summary.outward_class_counts?.[cat.key] || 0}</span>
                </div>
            `;
            breakdownGrid.appendChild(item);
        });
    }

    // -------------------------------------------------------------
    // PROCESS ANOTHER VIDEO
    // -------------------------------------------------------------
    processAnotherBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        setupSection.classList.remove('hidden');
        setupSection.scrollIntoView({ behavior: 'smooth' });
    });
});
