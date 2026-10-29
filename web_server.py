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

# Compute Device Initialization (Universal GPU / CPU Detection)
def init_compute_device():
    has_cuda = False
    device = "cpu"
    device_name = "CPU"
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            if torch.cuda.device_count() > 0:
                has_cuda = True
                device = "cuda:0"
                device_name = torch.cuda.get_device_name(0)
                torch.backends.cudnn.benchmark = True
                print(f"[*] Compute Device: {device} ({device_name}) [CUDA Active, cuDNN Benchmark Enabled]")
        except Exception as e:
            print(f"[!] Warning: CUDA initialization error: {e}. Falling back to CPU mode.")
            has_cuda = False
            device = "cpu"
            device_name = "CPU"

    if not has_cuda:
        try:
            threads = min(8, max(1, os.cpu_count() or 4))
            torch.set_num_threads(threads)
            print(f"[*] Compute Device: CPU ({threads} Multi-Threading Cores Active)")
        except Exception:
            pass
    return has_cuda, device, device_name

HAS_CUDA, DEVICE, DEVICE_NAME = init_compute_device()


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


def draw_frame_hud(frame, counter, fps_curr):
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
    if HAS_CUDA:
        dev_text = f"GPU: {DEVICE_NAME} | {fps_curr:.0f} FPS"
        dev_color = (6, 182, 212)
    else:
        dev_text = f"CPU MODE (No CUDA) | {fps_curr:.0f} FPS"
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

        # Load YOLO model and optimize for RTX 4070 SUPER
        model = YOLO(WEIGHTS_PATH)
        if HAS_CUDA:
            try:
                model.to(DEVICE)
                try:
                    model.model.half()
                    dummy = np.zeros((384, 640, 3), dtype=np.uint8)
                    model.predict(dummy, device=DEVICE, verbose=False, imgsz=640)
                except Exception:
                    # Fallback to FP32 for GPUs that don't support half precision
                    model.model.float()
            except Exception:
                pass

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

            # YOLO tracking inference with CUDA Tensor Cores + ByteTrack (high-speed GPU tracking)
            track_kwargs = {
                "source": frame,
                "persist": True,
                "verbose": False,
                "device": DEVICE,
                "conf": conf_thresh,
                "iou": 0.5,
                "imgsz": 640,
                "tracker": "bytetrack.yaml"
            }
            results = model.track(**track_kwargs)[0]

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

            # Draw HUD card with true live FPS
            frame = draw_frame_hud(frame, counter, live_fps)

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
            "device_name": DEVICE_NAME,
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
            "events": []
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
    """Returns real-time GPU / CUDA / CPU compute information."""
    has_cuda = torch.cuda.is_available()
    vram_free_gb = 0.0
    vram_total_gb = 0.0
    if has_cuda:
        try:
            free_b, tot_b = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_b / (1024 ** 3), 2)
            vram_total_gb = round(tot_b / (1024 ** 3), 2)
        except Exception:
            pass

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
