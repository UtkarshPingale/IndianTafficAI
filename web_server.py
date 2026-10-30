"""
Indian Vehicle Detection & Virtual Line Counter - Web Dashboard Server
Powered by Flask, PyTorch CUDA (RTX 4070 SUPER), Ultralytics YOLOv11s, and OpenPyXL.
"""

import os
import sys
import math
import time
import uuid
import threading
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
import imageio_ffmpeg

from vehicle_classes import map_class_name, get_category_color
from line_counter import LineCounter
from excel_reporter import generate_traffic_excel

import numpy as np
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Determine runtime environment paths (frozen .exe vs standard python)
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR

STATIC_DIR = os.path.join(RESOURCE_DIR, "static")
TEMPLATES_DIR = os.path.join(RESOURCE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.json = NumpyJSONProvider(app)


def to_serializable(obj):
    """Recursively converts NumPy types into standard Python serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return [to_serializable(x) for x in obj]
    return obj

UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Locate weights file
MODELS_DIR = os.path.join(APP_DIR, "models")
if not os.path.exists(os.path.join(MODELS_DIR, "yolo11s_indian_traffic.pt")):
    MODELS_DIR = os.path.join(RESOURCE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolo11s_indian_traffic.pt")

# Locate sample video
SAMPLE_VIDEO = os.path.join(APP_DIR, "compressed_output_TrafficPolice.mp4")
if not os.path.exists(SAMPLE_VIDEO):
    SAMPLE_VIDEO = os.path.join(RESOURCE_DIR, "compressed_output_TrafficPolice.mp4")

# In-memory job state store
jobs_lock = threading.Lock()
jobs = {}

# Distributed Client GPU Worker Fleet Store
gpu_workers_lock = threading.Lock()
gpu_workers = {}  # {worker_id: {info, last_heartbeat, status, current_job_id}}
pending_worker_jobs = []  # list of job_ids queued for remote GPU workers

def get_available_worker():
    """Returns an active, idle GPU worker if one has checked in recently (< 15 seconds)."""
    now = time.time()
    with gpu_workers_lock:
        for w_id, w in list(gpu_workers.items()):
            if now - w.get("last_heartbeat", 0) > 15:
                w["status"] = "offline"
            elif w.get("status") == "idle":
                return w
    return None

def get_active_workers_list():
    """Returns list of currently active connected client GPU workers."""
    now = time.time()
    active = []
    with gpu_workers_lock:
        for w_id, w in list(gpu_workers.items()):
            if now - w.get("last_heartbeat", 0) <= 15:
                active.append({
                    "id": w_id,
                    "name": w.get("name", "Client GPU Worker"),
                    "gpu_name": w.get("gpu_name", "NVIDIA GPU"),
                    "vram_gb": w.get("vram_gb", "N/A"),
                    "cuda_version": w.get("cuda_version", "Active"),
                    "status": w.get("status", "idle"),
                    "hostname": w.get("hostname", "Remote-PC"),
                    "jobs_completed": w.get("jobs_completed", 0),
                    "last_seen_sec": round(now - w.get("last_heartbeat", 0), 1)
                })
    return active

# Compute Device Initialization (Universal GPU / CPU Detection with Kernel Execution Verification)
def init_compute_device():
    has_cuda = False
    device = "cpu"
    device_name = "CPU (Multi-Core)"
    supports_fp16 = False
    vram_free = 0.0
    vram_total = 0.0

    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            if torch.cuda.device_count() > 0:
                raw_name = torch.cuda.get_device_name(0)

                # CRITICAL TEST: Execute matrix math on device to verify kernel availability
                # This catches "CUDA error: no kernel image is available for execution on the device"
                test_t = torch.zeros((1, 1), device="cuda:0")
                test_res = (test_t + 1.0).cpu()
                del test_t, test_res

                # Test FP16 support on this specific GPU
                try:
                    t_half = torch.zeros((1, 1), dtype=torch.float16, device="cuda:0")
                    res_half = (t_half + 1.0).cpu()
                    del t_half, res_half
                    supports_fp16 = True
                except Exception:
                    supports_fp16 = False

                try:
                    free_b, tot_b = torch.cuda.mem_get_info(0)
                    vram_free = round(free_b / (1024 ** 3), 2)
                    vram_total = round(tot_b / (1024 ** 3), 2)
                except Exception:
                    pass

                # Test cuDNN / convolution execution on CUDA
                try:
                    with torch.no_grad():
                        test_conv = torch.nn.functional.conv2d(
                            torch.zeros((1, 1, 3, 3), device="cuda:0"),
                            torch.zeros((1, 1, 3, 3), device="cuda:0")
                        )
                        del test_conv
                    torch.backends.cudnn.benchmark = True
                except Exception as cudnn_err:
                    print(f"[!] Warning: cuDNN sublibrary issue ({cudnn_err}). Disabling cuDNN, keeping CUDA active.")
                    torch.backends.cudnn.enabled = False

                has_cuda = True
                device = "cuda:0"
                device_name = raw_name
                print(f"[*] Compute Device: {device} ({device_name}) [CUDA Active, FP16: {supports_fp16}, VRAM: {vram_free}/{vram_total} GB]")
        except Exception as e:
            err_msg = str(e)
            print(f"[!] Warning: NVIDIA GPU detected, but cannot execute CUDA 12.4 kernels ({err_msg}).")
            print("[*] Automatically falling back to optimized CPU mode for universal compatibility!")
            has_cuda = False
            device = "cpu"
            device_name = "CPU Mode (GPU Compute Architecture Mismatch)"
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if not has_cuda:
        try:
            threads = min(8, max(1, os.cpu_count() or 4))
            torch.set_num_threads(threads)
            print(f"[*] Compute Device: CPU ({threads} Multi-Threading Cores Active)")
        except Exception:
            pass

    return has_cuda, device, device_name, supports_fp16, vram_free, vram_total

HAS_CUDA, DEVICE, DEVICE_NAME, SUPPORTS_FP16, VRAM_FREE_GB, VRAM_TOTAL_GB = init_compute_device()


def get_video_info(video_path):
    """Extracts width, height, fps, and total frames of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": w, "height": h, "fps": round(fps, 2), "total_frames": total_frames}


def extract_first_frame(video_path, output_jpg_path):
    """Saves first frame of video as JPEG for canvas line placement."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        cv2.imwrite(output_jpg_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return True
    return False


def draw_frame_hud(frame, counter, fps_curr, active_device=DEVICE):
    """Draws sleek HUD card on the video frame using fast in-place ROI blending."""
    card_w = 320
    card_h = 240
    margin = 15

    h_f, w_f = frame.shape[:2]
    x1, y1 = margin, margin
    x2, y2 = margin + card_w, margin + card_h

    # Fast in-place ROI blending without copying the whole video frame
    if x2 <= w_f and y2 <= h_f:
        roi = frame[y1:y2, x1:x2]
        card = np.full_like(roi, (15, 23, 42), dtype=np.uint8)
        cv2.addWeighted(card, 0.82, roi, 0.18, 0, roi)
        frame[y1:y2, x1:x2] = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 65, 85), 1)

    cv2.putText(frame, "INDIAN TRAFFIC COUNTER", (margin + 12, margin + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Dynamic device indicator based on actual runtime hardware
    if HAS_CUDA and str(active_device).startswith("cuda"):
        dev_text = f"GPU: {DEVICE_NAME} | {fps_curr:.0f} FPS"
        dev_color = (6, 182, 212)
    else:
        dev_text = f"CPU MODE | {fps_curr:.0f} FPS"
        dev_color = (0, 140, 255)

    cv2.putText(frame, dev_text, (margin + 12, margin + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, dev_color, 1, cv2.LINE_AA)
    cv2.line(frame, (margin + 12, margin + 52), (margin + card_w - 12, margin + 52), (51, 65, 85), 1)

    tot_str = f"TOTAL COUNT: {counter.total_count}"
    cv2.putText(frame, tot_str, (margin + 12, margin + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

    in_cnt = counter.direction_counts.get("Inward", 0)
    out_cnt = counter.direction_counts.get("Outward", 0)
    cv2.putText(frame, f"INWARD: {in_cnt} | OUTWARD: {out_cnt}", (margin + 12, margin + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 230, 180), 1, cv2.LINE_AA)

    # Class breakdown
    y = margin + 116
    for cls_name, cnt in counter.class_counts.most_common(5):
        color = get_category_color(cls_name)
        cv2.circle(frame, (margin + 16, y - 4), 4, color, -1)
        cv2.putText(frame, f"{cls_name}: {cnt}", (margin + 28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        y += 20
    return frame


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample-video", methods=["POST"])
def select_sample_video():
    """Selects the pre-packaged sample Indian traffic video."""
    if not os.path.exists(SAMPLE_VIDEO):
        return jsonify({"status": "error", "message": "Sample video not found"}), 404

    info = get_video_info(SAMPLE_VIDEO)
    if not info:
        return jsonify({"status": "error", "message": "Cannot open sample video"}), 500

    frame_path = os.path.join(UPLOADS_DIR, "sample_preview_frame.jpg")
    extract_first_frame(SAMPLE_VIDEO, frame_path)

    return jsonify({
        "status": "ok",
        "video_path": SAMPLE_VIDEO,
        "filename": os.path.basename(SAMPLE_VIDEO),
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "total_frames": info["total_frames"],
        "frame_url": "/api/frame/sample_preview_frame.jpg"
    })


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Uploads a user video file."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    saved_filename = f"user_upload_{unique_id}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)
    file.save(saved_path)

    info = get_video_info(saved_path)
    if not info:
        return jsonify({"status": "error", "message": "Uploaded file is not a valid video"}), 400

    frame_name = f"frame_{unique_id}.jpg"
    frame_path = os.path.join(UPLOADS_DIR, frame_name)
    extract_first_frame(saved_path, frame_path)

    return jsonify({
        "status": "ok",
        "video_path": saved_path,
        "filename": file.filename,
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "total_frames": info["total_frames"],
        "frame_url": f"/api/frame/{frame_name}"
    })


@app.route("/api/frame/<path:filename>")
def serve_frame(filename):
    """Serves preview road frames."""
    path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "Frame not found", 404


def run_video_job(job_id, video_path, line_coords, conf_thresh):
    """Background worker processing video on RTX 4070 SUPER."""
    try:
        t_start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"Failed to open video {video_path}"
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Auto-scale high-resolution videos (4K / 1440p) down to 1080p for real-time processing
        max_dim = max(w, h)
        scale = 1920.0 / max_dim if max_dim > 1920 else 1.0
        proc_w = int(w * scale)
        proc_h = int(h * scale)

        p1 = (int(line_coords[0] * scale), int(line_coords[1] * scale))
        p2 = (int(line_coords[2] * scale), int(line_coords[3] * scale))
        counter = LineCounter(pt1=p1, pt2=p2)

        # Load YOLO model and optimize for compute hardware
        model = YOLO(WEIGHTS_PATH)
        current_device = DEVICE
        if HAS_CUDA and str(current_device).startswith("cuda"):
            try:
                model.to(current_device)
                dummy = np.zeros((384, 640, 3), dtype=np.uint8)
                if SUPPORTS_FP16:
                    try:
                        model.model.half()
                        model.predict(dummy, device=current_device, verbose=False, imgsz=640)
                    except Exception:
                        model.model.float()
                        model.predict(dummy, device=current_device, verbose=False, imgsz=640)
                else:
                    model.model.float()
                    model.predict(dummy, device=current_device, verbose=False, imgsz=640)
            except Exception as e:
                print(f"[!] Warning: CUDA execution failed on this GPU ({e}). Seamlessly switching to CPU mode...")
                current_device = "cpu"
                try:
                    model.to("cpu")
                    model.model.float()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            current_device = "cpu"

        # Output paths
        web_video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
        excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
        csv_path = os.path.join(OUTPUTS_DIR, f"log_{job_id}.csv")

        ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
        frame_bytes = proc_w * proc_h * 3

        # Hardware-accelerated input video decoding via NVIDIA NVDEC
        pipe_in = None
        first_frame = None
        if HAS_CUDA:
            try:
                in_cmd = [
                    ff_exe,
                    "-hwaccel", "cuda",
                    "-i", video_path,
                    "-vf", f"scale={proc_w}:{proc_h}",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-v", "error",
                    "-"
                ]
                pipe_in = subprocess.Popen(
                    in_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=frame_bytes * 4
                )
                first_raw = pipe_in.stdout.read(frame_bytes)
                if first_raw and len(first_raw) == frame_bytes:
                    first_frame = np.frombuffer(first_raw, dtype=np.uint8).reshape((proc_h, proc_w, 3)).copy()
                else:
                    pipe_in.kill()
                    pipe_in = None
            except Exception:
                pipe_in = None

        # Hardware-accelerated direct FFmpeg pipe (prefers NVIDIA NVENC GPU encoder)
        pipe_proc = None
        cv_writer = None

        if HAS_CUDA:
            try:
                nvenc_cmd = [
                    ff_exe, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{proc_w}x{proc_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(src_fps),
                    "-i", "-",
                    "-c:v", "h264_nvenc",
                    "-preset", "p1",
                    "-pix_fmt", "yuv420p",
                    web_video_path
                ]
                pipe_proc = subprocess.Popen(
                    nvenc_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(0.08)
                if pipe_proc.poll() is not None:
                    # NVENC is unsupported on this GPU or driver, fall back to CPU libx264
                    pipe_proc = None
            except Exception:
                pipe_proc = None

        if pipe_proc is None:
            try:
                cpu_cmd = [
                    ff_exe, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{proc_w}x{proc_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(src_fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    web_video_path
                ]
                pipe_proc = subprocess.Popen(
                    cpu_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                raw_video_path = os.path.join(OUTPUTS_DIR, f"raw_{job_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                cv_writer = cv2.VideoWriter(raw_video_path, fourcc, src_fps, (proc_w, proc_h))

        frame_idx = 0
        t_loop_start = time.time()
        last_progress_update = time.time()
        fps_window_start = time.time()
        fps_window_frames = 0
        live_fps = src_fps

        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
            elif pipe_in and pipe_in.stdout:
                raw_bytes = pipe_in.stdout.read(frame_bytes)
                if not raw_bytes or len(raw_bytes) < frame_bytes:
                    break
                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((proc_h, proc_w, 3)).copy()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if scale < 1.0:
                    frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            frame_idx += 1

            # YOLO tracking inference with auto-fallback to CPU if CUDA fails
            track_kwargs = {
                "source": frame,
                "persist": True,
                "verbose": False,
                "device": current_device,
                "conf": conf_thresh,
                "iou": 0.5,
                "imgsz": 640,
                "tracker": "bytetrack.yaml"
            }
            try:
                results = model.track(**track_kwargs)[0]
            except Exception as e:
                err_str = str(e).lower()
                # If cuDNN engine loading failed, try turning off cuDNN first before dropping CUDA
                if ("cudnn" in err_str or "sublibrary" in err_str) and torch.backends.cudnn.enabled:
                    print(f"[!] cuDNN issue encountered ({e}). Disabling cuDNN and retrying on CUDA...")
                    torch.backends.cudnn.enabled = False
                    try:
                        results = model.track(**track_kwargs)[0]
                    except Exception as e2:
                        e = e2
                        err_str = str(e).lower()
                if ("cuda" in err_str or "kernel" in err_str or "device" in err_str or "out of memory" in err_str or "cudnn" in err_str or "sublibrary" in err_str) and current_device != "cpu":
                    print(f"[!] Hardware CUDA/cuDNN error on frame {frame_idx} ({e}). Seamlessly switching to CPU mode...")
                    current_device = "cpu"
                    track_kwargs["device"] = "cpu"
                    try:
                        model.to("cpu")
                        model.model.float()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    results = model.track(**track_kwargs)[0]
                else:
                    raise e

            active_ids = set()
            boxes = results.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for bbox, track_id, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                    active_ids.add(int(track_id))
                    raw_name = model.names.get(cls_id, "Vehicle")
                    category = map_class_name(raw_name)
                    color = get_category_color(category)

                    # Update line crossing
                    counter.update_vehicle(
                        track_id=track_id,
                        bbox=bbox,
                        raw_class_name=raw_name,
                        conf=float(conf),
                        frame_idx=frame_idx
                    )

                    # Draw bounding box and label
                    bx1, by1, bx2, by2 = bbox
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)

                    label = f"#{track_id} {category} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                    cv2.rectangle(frame, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(0, by1)), color, -1)
                    cv2.putText(frame, label, (bx1 + 3, max(0, by1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw trajectory trails for active vehicles & virtual line
            frame = counter.draw_trajectories(frame, active_ids=active_ids)
            frame = counter.draw_line(frame)

            # Rolling FPS window calculation
            now = time.time()
            elapsed_time = now - t_loop_start
            fps_window_frames += 1
            if now - fps_window_start >= 0.4:
                live_fps = fps_window_frames / (now - fps_window_start)
                fps_window_start = now
                fps_window_frames = 0

            # Draw HUD card with true live FPS and active device indicator
            frame = draw_frame_hud(frame, counter, live_fps, active_device=current_device)

            # Output frame
            if pipe_proc and pipe_proc.stdin:
                try:
                    pipe_proc.stdin.write(frame.tobytes())
                except Exception:
                    pass
            elif cv_writer:
                cv_writer.write(frame)

            # Update progress status
            if now - last_progress_update > 0.35 or frame_idx == total_frames:
                last_progress_update = now
                pct = int((frame_idx / max(total_frames, 1)) * 100)
                rem_frames = max(0, total_frames - frame_idx)
                eta = (rem_frames / live_fps) if live_fps > 0 else 0

                with jobs_lock:
                    jobs[job_id]["current_frame"] = frame_idx
                    jobs[job_id]["percent"] = min(pct, 98)
                    jobs[job_id]["fps"] = live_fps
                    jobs[job_id]["elapsed_sec"] = elapsed_time
                    jobs[job_id]["eta_sec"] = eta
                    jobs[job_id]["total_count"] = counter.total_count
                    jobs[job_id]["active_counts"] = dict(counter.class_counts)

        if pipe_in:
            try:
                pipe_in.kill()
            except Exception:
                pass

        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if pipe_proc:
            if pipe_proc.stdin:
                try:
                    pipe_proc.stdin.close()
                except Exception:
                    pass
            pipe_proc.wait()
        elif cv_writer:
            cv_writer.release()
            # Transcode fallback
            raw_video_path = os.path.join(OUTPUTS_DIR, f"raw_{job_id}.mp4")
            if os.path.exists(raw_video_path):
                cmd = [
                    ff_exe, "-y",
                    "-i", raw_video_path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    web_video_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                try:
                    os.remove(raw_video_path)
                except OSError:
                    pass

        t_total = time.time() - t_loop_start
        avg_fps = frame_idx / t_total if t_total > 0 else 0

        # Build Summary Data
        summary_data = {
            "total_count": counter.total_count,
            "inward_count": counter.direction_counts.get("Inward", 0),
            "outward_count": counter.direction_counts.get("Outward", 0),
            "class_counts": dict(counter.class_counts),
            "inward_class_counts": {},
            "outward_class_counts": {},
            "video_name": os.path.basename(video_path),
            "fps": avg_fps,
            "total_frames": frame_idx,
            "device_name": DEVICE_NAME if current_device != "cpu" else "CPU Mode",
            "process_time": t_total
        }

        for ev in counter.events:
            c = ev["class"]
            d = ev["direction"]
            if d == "Inward":
                summary_data["inward_class_counts"][c] = summary_data["inward_class_counts"].get(c, 0) + 1
            elif d == "Outward":
                summary_data["outward_class_counts"][c] = summary_data["outward_class_counts"].get(c, 0) + 1

        # Generate Styled Excel & CSV
        generate_traffic_excel(counter.events, summary_data, excel_path)
        
        # Save CSV
        df = pd.DataFrame(counter.events)
        df.to_csv(csv_path, index=False)

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["percent"] = 100
            jobs[job_id]["total_frames"] = frame_idx
            jobs[job_id]["fps"] = avg_fps
            jobs[job_id]["summary"] = summary_data
            jobs[job_id]["events"] = counter.events
            jobs[job_id]["web_video_path"] = web_video_path
            jobs[job_id]["excel_path"] = excel_path
            jobs[job_id]["csv_path"] = csv_path

    except Exception as e:
        if "pipe_in" in locals() and pipe_in:
            try:
                pipe_in.kill()
            except Exception:
                pass
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


@app.route("/api/process", methods=["POST"])
def start_process():
    """Launches video processing."""
    data = request.get_json() or {}
    video_source = data.get("video_source")
    if not video_source or not os.path.exists(video_source):
        return jsonify({"status": "error", "message": "Invalid video source"}), 400

    coords_str = data.get("line", "100,450,1180,450")
    try:
        parts = [int(p.strip()) for p in coords_str.split(",")]
        if len(parts) != 4:
            raise ValueError()
    except Exception:
        parts = [100, 450, 1180, 450]

    conf_thresh = float(data.get("conf", 0.35))
    job_id = str(uuid.uuid4())[:8]

    info = get_video_info(video_source)
    total_frames = info["total_frames"] if info else 100

    # Check if an active Client GPU Worker is connected
    available_worker = get_available_worker()
    prefer_worker = data.get("prefer_worker", True)

    if available_worker and prefer_worker:
        with jobs_lock:
            jobs[job_id] = {
                "status": "processing",
                "percent": 0,
                "current_frame": 0,
                "total_frames": total_frames,
                "fps": 0.0,
                "elapsed_sec": 0.0,
                "eta_sec": 0.0,
                "total_count": 0,
                "active_counts": {},
                "summary": None,
                "events": [],
                "assigned_worker": available_worker["id"],
                "worker_name": available_worker.get("name"),
                "device_name": f"{available_worker.get('gpu_name')} ({available_worker.get('name')})",
                "video_source": video_source,
                "line_coords": parts,
                "conf_thresh": conf_thresh
            }

        with gpu_workers_lock:
            available_worker["status"] = "busy"
            available_worker["current_job_id"] = job_id
            pending_worker_jobs.append(job_id)

        print(f"[*] Dispatched Job {job_id} to Client GPU Worker: {available_worker.get('name')} [{available_worker.get('gpu_name')}]")
    else:
        with jobs_lock:
            jobs[job_id] = {
                "status": "processing",
                "percent": 0,
                "current_frame": 0,
                "total_frames": total_frames,
                "fps": 0.0,
                "elapsed_sec": 0.0,
                "eta_sec": 0.0,
                "total_count": 0,
                "active_counts": {},
                "summary": None,
                "events": [],
                "assigned_worker": "local_server",
                "worker_name": "Server Host",
                "device_name": DEVICE_NAME
            }

        t = threading.Thread(
            target=run_video_job,
            args=(job_id, video_source, parts, conf_thresh),
            daemon=True
        )
        t.start()

    return jsonify({"status": "ok", "job_id": job_id})


@app.route("/api/progress/<job_id>")
def get_progress(job_id):
    """Returns real-time progress for active job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        
        # Return lightweight, clean progress dict (excludes heavy events)
        resp = {
            "status": str(job.get("status", "unknown")),
            "percent": int(job.get("percent", 0)),
            "current_frame": int(job.get("current_frame", 0)),
            "total_frames": int(job.get("total_frames", 0)),
            "fps": float(round(float(job.get("fps", 0.0)), 1)),
            "elapsed_sec": float(round(float(job.get("elapsed_sec", 0.0)), 1)),
            "eta_sec": float(round(float(job.get("eta_sec", 0.0)), 1)),
            "total_count": int(job.get("total_count", 0)),
            "active_counts": {str(k): int(v) for k, v in job.get("active_counts", {}).items()},
            "error": str(job.get("error")) if job.get("error") else None
        }
        return jsonify(resp)


@app.route("/api/results/<job_id>")
def get_results(job_id):
    """Returns final summary and events array for Excel table preview."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        if job["status"] != "completed":
            return jsonify({"status": job["status"]}), 202

        return jsonify({
            "status": "completed",
            "summary": to_serializable(job.get("summary", {})),
            "events": to_serializable(job.get("events", []))
        })


@app.route("/api/video/<job_id>")
def serve_video(job_id):
    """Streams the browser-compatible H.264 video."""
    video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")
    return "Video not found", 404


@app.route("/api/download/excel/<job_id>")
def download_excel(job_id):
    """Downloads styled multi-sheet Excel file."""
    excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
    if os.path.exists(excel_path):
        return send_file(
            excel_path,
            as_attachment=True,
            download_name=f"Traffic_Report_{job_id}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return "Excel file not found", 404


@app.route("/api/download/csv/<job_id>")
def download_csv(job_id):
    """Downloads raw CSV log."""
    csv_path = os.path.join(OUTPUTS_DIR, f"log_{job_id}.csv")
    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            as_attachment=True,
            download_name=f"Traffic_Log_{job_id}.csv",
            mimetype="text/csv"
        )
    return "CSV not found", 404


@app.route("/api/hardware-status")
def hardware_status():
    """Returns real-time GPU / CUDA / CPU compute information, including connected Client GPU Workers."""
    has_cuda = DEVICE.startswith("cuda")
    vram_free_gb = 0.0
    vram_total_gb = 0.0
    device_name = DEVICE_NAME
    if has_cuda and torch.cuda.is_available():
        try:
            free_b, tot_b = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_b / (1024 ** 3), 2)
            vram_total_gb = round(tot_b / (1024 ** 3), 2)
        except Exception:
            pass

    active_workers = get_active_workers_list()
    compute_mode = "local_gpu" if has_cuda else "cpu"
    
    # If client GPU workers are online, highlight client GPU acceleration!
    if active_workers:
        compute_mode = "client_gpu_fleet"
        primary = active_workers[0]
        device_name = f"{primary['gpu_name']} ({primary['name']})"
        has_cuda = True

    return jsonify({
        "has_cuda": has_cuda,
        "device": DEVICE,
        "device_name": device_name,
        "compute_mode": compute_mode,
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "supports_fp16": SUPPORTS_FP16,
        "vram_free_gb": vram_free_gb,
        "vram_total_gb": vram_total_gb,
        "connected_workers": active_workers,
        "worker_count": len(active_workers)
    })


# =========================================================================
# DISTRIBUTED CLIENT GPU WORKER FLEET API
# Allows remote client PCs with NVIDIA GPUs (e.g. RTX 5070) to process jobs
# =========================================================================

@app.route("/api/worker/register", methods=["POST"])
def worker_register():
    """Client GPU worker registration endpoint."""
    data = request.get_json() or {}
    w_id = data.get("worker_id") or str(uuid.uuid4())[:8]
    now = time.time()
    
    with gpu_workers_lock:
        gpu_workers[w_id] = {
            "id": w_id,
            "name": data.get("name") or f"Worker-{w_id}",
            "hostname": data.get("hostname", "Client-PC"),
            "gpu_name": data.get("gpu_name", "NVIDIA GPU"),
            "cuda_version": data.get("cuda_version", "Active"),
            "vram_gb": data.get("vram_gb", "N/A"),
            "last_heartbeat": now,
            "status": "idle",
            "current_job_id": None,
            "jobs_completed": 0
        }
    
    print(f"[+] Client GPU Worker Registered: {gpu_workers[w_id]['name']} [{gpu_workers[w_id]['gpu_name']}]")
    return jsonify({"status": "ok", "worker_id": w_id, "message": "Worker registered successfully"})


@app.route("/api/worker/heartbeat", methods=["POST"])
def worker_heartbeat():
    """Client GPU worker heartbeat."""
    data = request.get_json() or {}
    w_id = data.get("worker_id")
    if not w_id:
        return jsonify({"status": "error", "message": "worker_id required"}), 400
    
    with gpu_workers_lock:
        if w_id in gpu_workers:
            gpu_workers[w_id]["last_heartbeat"] = time.time()
            if "status" in data and gpu_workers[w_id]["status"] != "busy":
                gpu_workers[w_id]["status"] = data["status"]
            return jsonify({"status": "ok", "ack": True})
        else:
            return jsonify({"status": "need_registration"}), 404


@app.route("/api/worker/job/poll", methods=["GET"])
def worker_poll_job():
    """Worker checks if there is a pending job assigned to it."""
    w_id = request.args.get("worker_id")
    if not w_id:
        return jsonify({"status": "error", "message": "worker_id required"}), 400

    with gpu_workers_lock:
        if w_id not in gpu_workers:
            return jsonify({"status": "need_registration"}), 404
        gpu_workers[w_id]["last_heartbeat"] = time.time()

        job_to_assign = None
        if pending_worker_jobs:
            job_to_assign = pending_worker_jobs.pop(0)

    if not job_to_assign:
        return jsonify({"status": "idle"})

    with jobs_lock:
        job = jobs.get(job_to_assign)
        if not job or job.get("status") == "completed":
            return jsonify({"status": "idle"})

        job["status"] = "processing"
        job["assigned_worker"] = w_id
        
        return jsonify({
            "status": "job_available",
            "job_id": job_to_assign,
            "video_url": f"/api/worker/job/{job_to_assign}/video",
            "line_coords": job.get("line_coords", [100, 450, 1180, 450]),
            "conf_thresh": job.get("conf_thresh", 0.35),
            "video_filename": os.path.basename(job.get("video_source", "video.mp4"))
        })


@app.route("/api/worker/job/<job_id>/video")
def worker_download_video(job_id):
    """Worker downloads raw video for local GPU processing."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return "Job not found", 404
        video_path = job.get("video_source")
        if not video_path or not os.path.exists(video_path):
            return "Video file missing", 404
        return send_file(video_path, mimetype="video/mp4")


@app.route("/api/worker/job/<job_id>/progress", methods=["POST"])
def worker_report_progress(job_id):
    """Worker reports live frame processing progress and vehicle counts."""
    data = request.get_json() or {}
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404

        job["percent"] = data.get("percent", job.get("percent", 0))
        job["current_frame"] = data.get("current_frame", job.get("current_frame", 0))
        job["total_frames"] = data.get("total_frames", job.get("total_frames", 100))
        job["fps"] = data.get("fps", job.get("fps", 0.0))
        job["elapsed_sec"] = data.get("elapsed_sec", 0.0)
        job["eta_sec"] = data.get("eta_sec", 0.0)
        job["total_count"] = data.get("total_count", job.get("total_count", 0))
        job["active_counts"] = data.get("active_counts", {})
        job["status"] = "processing"

    return jsonify({"status": "ok"})


@app.route("/api/worker/job/<job_id>/complete", methods=["POST"])
def worker_job_complete(job_id):
    """Worker submits final processed video and crossing events."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404

    # Save uploaded processed video
    web_video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
    if "video" in request.files:
        video_file = request.files["video"]
        video_file.save(web_video_path)
    
    # Parse events & summary metadata
    events = []
    summary_data = {}
    if "events" in request.form:
        try:
            events = json.loads(request.form["events"])
        except Exception:
            events = []
    if "summary" in request.form:
        try:
            summary_data = json.loads(request.form["summary"])
        except Exception:
            summary_data = {}

    excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
    csv_path = os.path.join(OUTPUTS_DIR, f"log_{job_id}.csv")

    try:
        generate_traffic_excel(events, summary_data, excel_path)
    except Exception as e:
        print(f"[!] Excel generation warning: {e}")

    try:
        df = pd.DataFrame(events)
        df.to_csv(csv_path, index=False)
    except Exception:
        pass

    with jobs_lock:
        job["status"] = "completed"
        job["percent"] = 100
        job["summary"] = summary_data
        job["events"] = events
        job["web_video_path"] = web_video_path
        job["excel_path"] = excel_path
        job["csv_path"] = csv_path
        assigned_worker = job.get("assigned_worker")

    with gpu_workers_lock:
        if assigned_worker and assigned_worker in gpu_workers:
            gpu_workers[assigned_worker]["status"] = "idle"
            gpu_workers[assigned_worker]["current_job_id"] = None
            gpu_workers[assigned_worker]["jobs_completed"] += 1

    print(f"[OK] Job {job_id} successfully completed by Client GPU Worker! Results saved.")
    return jsonify({"status": "ok", "job_id": job_id})


@app.route("/api/worker/job/<job_id>/error", methods=["POST"])
def worker_job_error(job_id):
    """Worker reports a failure during processing."""
    data = request.get_json() or {}
    err = data.get("error", "Unknown worker error")
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Worker Error: {err}"
            assigned_worker = jobs[job_id].get("assigned_worker")

    with gpu_workers_lock:
        if assigned_worker and assigned_worker in gpu_workers:
            gpu_workers[assigned_worker]["status"] = "idle"
            gpu_workers[assigned_worker]["current_job_id"] = None

    return jsonify({"status": "ok"})


@app.route("/api/workers", methods=["GET"])
def get_workers():
    """Returns all active connected GPU workers."""
    return jsonify({
        "status": "ok",
        "workers": get_active_workers_list()
    })


@app.route("/api/download/worker-client", methods=["GET"])
def download_worker_client_zip():
    """Generates and serves the lightweight Client GPU Worker package (.zip ~20 MB)."""
    zip_path = os.path.join(APP_DIR, "IndianTrafficAI_GPU_Worker.zip")
    if not os.path.exists(zip_path):
        import build_worker_package
        build_worker_package.build_package(zip_path)
    
    if os.path.exists(zip_path):
        return send_file(
            zip_path,
            as_attachment=True,
            download_name="IndianTrafficAI_GPU_Worker.zip",
            mimetype="application/zip"
        )
    return "Worker package not found", 404

# --- Auto-Update Endpoints ---
update_progress = {"percent": 0, "status": "idle", "error": None}

@app.route("/api/update/status")
def update_status():
    """Returns local and cached/remote version details."""
    import auto_updater
    cfg = auto_updater.load_config()
    return jsonify({
        "status": "ok",
        "current_version": cfg.get("version", "1.0.0"),
        "release_date": cfg.get("release_date", ""),
        "update_manifest_url": cfg.get("update_manifest_url", ""),
        "auto_check_on_launch": cfg.get("auto_check_on_launch", True)
    })

@app.route("/api/update/check", methods=["GET", "POST"])
def check_update_route():
    """Queries remote manifest to check for new updates."""
    import auto_updater
    res = auto_updater.check_for_update()
    return jsonify(res)

@app.route("/api/update/apply", methods=["POST"])
def apply_update_route():
    """Downloads remote update, executes detached updater batch, and restarts app."""
    global update_progress
    import auto_updater
    data = request.get_json() or {}
    download_url = data.get("download_url")

    if not download_url:
        check_res = auto_updater.check_for_update()
        download_url = check_res.get("download_url")

    if not download_url:
        return jsonify({"status": "error", "message": "No download URL available for update."}), 400

    def _worker():
        global update_progress
        try:
            update_progress["status"] = "downloading"
            update_progress["percent"] = 0
            
            def cb(pct, down, tot):
                update_progress["percent"] = pct
                
            zip_path = auto_updater.download_update(download_url, progress_callback=cb)
            update_progress["status"] = "extracting"
            extracted = auto_updater.extract_update_package(zip_path)
            update_progress["status"] = "restarting"
            time.sleep(1)
            auto_updater.apply_and_restart(extracted)
        except Exception as e:
            update_progress["status"] = "error"
            update_progress["error"] = str(e)

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "ok", "message": "Update process initiated."})

@app.route("/api/update/progress")
def update_progress_route():
    """Returns live update download & installation progress."""
    global update_progress
    return jsonify(update_progress)

@app.route("/api/update/config", methods=["POST"])
def update_config_route():
    """Saves customized update manifest or download URLs."""
    import auto_updater
    data = request.get_json() or {}
    cfg = auto_updater.load_config()
    if "update_manifest_url" in data:
        cfg["update_manifest_url"] = str(data["update_manifest_url"]).strip()
    if "auto_check_on_launch" in data:
        cfg["auto_check_on_launch"] = bool(data["auto_check_on_launch"])
    auto_updater.save_config(cfg)
    return jsonify({"status": "ok", "config": cfg})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"🚀 Indian Traffic AI Counter Web Dashboard Active!")
    print(f"📍 Local URL: http://localhost:{port}")
    print(f"⚡ GPU: {DEVICE_NAME} ({DEVICE})")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
