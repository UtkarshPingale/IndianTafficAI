"""
Indian Traffic AI - Client GPU Worker Node
Connects to the Central Web Server (on the host PC), registers the client's NVIDIA GPU
(e.g., RTX 5070 / RTX 4070 / any CUDA GPU), polls for video analysis jobs, and executes
high-throughput YOLOv11 tracking directly on local GPU Tensor Cores.
"""

import os
import sys
import time
import json
import uuid
import socket
import argparse
import threading
import requests
import cv2
import numpy as np
import torch

from vehicle_counter import VehicleCounterSystem

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(APP_DIR, "worker_config.json")
STAGING_DIR = os.path.join(APP_DIR, "worker_staging")
os.makedirs(STAGING_DIR, exist_ok=True)


def load_worker_config():
    """Loads worker configuration from disk or returns sensible defaults."""
    default_cfg = {
        "server_url": "http://127.0.0.1:5000",
        "worker_name": f"Worker-{socket.gethostname()}",
        "poll_interval_sec": 2.0,
        "heartbeat_interval_sec": 5.0
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                default_cfg.update(user_cfg)
        except Exception:
            pass
    return default_cfg


def save_worker_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[!] Warning: Could not save worker config: {e}")


def get_local_gpu_info():
    """Scans and verifies local client GPU hardware and CUDA execution capabilities."""
    has_cuda = torch.cuda.is_available()
    gpu_name = "CPU Mode"
    cuda_ver = "None"
    vram_free_gb = 0.0
    vram_total_gb = 0.0
    supports_fp16 = False

    if has_cuda:
        try:
            torch.cuda.init()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "Active"

            # Execute actual CUDA matrix test on client GPU
            test_t = torch.zeros((1, 1), device="cuda:0") + 1.0
            del test_t

            # Test cuDNN convolution execution
            try:
                with torch.no_grad():
                    test_conv = torch.nn.functional.conv2d(
                        torch.zeros((1, 1, 3, 3), device="cuda:0"),
                        torch.zeros((1, 1, 3, 3), device="cuda:0")
                    )
                    del test_conv
                torch.backends.cudnn.benchmark = True
            except Exception as e:
                print(f"[!] Warning: cuDNN issue on client GPU ({e}). Using standard CUDA kernels.")
                torch.backends.cudnn.enabled = False

            # Test FP16 support
            try:
                t_half = torch.zeros((1, 1), dtype=torch.float16, device="cuda:0") + 1.0
                del t_half
                supports_fp16 = True
            except Exception:
                supports_fp16 = False

            free_b, tot_b = torch.cuda.mem_get_info(0)
            vram_free_gb = round(free_b / (1024 ** 3), 2)
            vram_total_gb = round(tot_b / (1024 ** 3), 2)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[!] Warning: CUDA initialization error on client: {e}. Falling back to CPU mode.")
            has_cuda = False
            gpu_name = "CPU Mode (GPU Incompatible)"

    return {
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "cuda_version": cuda_ver,
        "vram_free_gb": vram_free_gb,
        "vram_total_gb": vram_total_gb,
        "supports_fp16": supports_fp16,
        "torch_version": torch.__version__,
        "hostname": socket.gethostname()
    }


class ClientGPUWorker:
    """Manages connection to Central Web Server and executes GPU tracking jobs."""

    def __init__(self, server_url, worker_name=None):
        self.server_url = server_url.rstrip("/")
        self.worker_id = f"worker-{socket.gethostname().lower()}-{str(uuid.uuid4())[:6]}"
        self.worker_name = worker_name or f"GPU-Node-{socket.gethostname()}"
        self.gpu_info = get_local_gpu_info()
        self.status = "idle"
        self.is_running = True
        self.jobs_completed = 0
        self.current_job_id = None
        self.weights_path = os.path.join(APP_DIR, "models", "yolo11s_indian_traffic.pt")

    def print_banner(self):
        print("=" * 72)
        print("      INDIAN TRAFFIC AI - DISTRIBUTED CLIENT GPU WORKER NODE     ")
        print("             High-Performance Remote Inference Engine             ")
        print("=" * 72)
        print(f"[*] Worker ID      : {self.worker_id}")
        print(f"[*] Worker Name    : {self.worker_name} ({self.gpu_info['hostname']})")
        print(f"[*] Central Server : {self.server_url}")
        if self.gpu_info["has_cuda"]:
            print(f"[+] Local Compute  : {self.gpu_info['gpu_name']}")
            print(f"[+] CUDA Engine    : {self.gpu_info['cuda_version']} (Active)")
            print(f"[+] VRAM Capacity  : {self.gpu_info['vram_total_gb']} GB ({self.gpu_info['vram_free_gb']} GB Free)")
            print(f"[+] Tensor Cores   : {'FP16 Enabled' if self.gpu_info['supports_fp16'] else 'Standard FP32'}")
        else:
            print(f"[!] Compute Mode   : CPU Multi-Threading (No NVIDIA GPU detected)")
        print("-" * 72)

    def register_with_server(self):
        """Registers this worker with the central server."""
        url = f"{self.server_url}/api/worker/register"
        payload = {
            "worker_id": self.worker_id,
            "name": self.worker_name,
            "hostname": self.gpu_info["hostname"],
            "gpu_name": self.gpu_info["gpu_name"],
            "cuda_version": self.gpu_info["cuda_version"],
            "vram_gb": f"{self.gpu_info['vram_free_gb']}/{self.gpu_info['vram_total_gb']} GB",
            "has_cuda": self.gpu_info["has_cuda"]
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                print(f"[OK] Successfully connected and registered with Central Server!")
                return True
            else:
                print(f"[!] Server returned status {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            print(f"[!] Connection failed to {url}: {e}")
            return False

    def heartbeat_loop(self):
        """Sends a keepalive ping every 5 seconds in a background thread."""
        url = f"{self.server_url}/api/worker/heartbeat"
        while self.is_running:
            try:
                requests.post(
                    url,
                    json={"worker_id": self.worker_id, "status": self.status},
                    timeout=5
                )
            except Exception:
                pass
            time.sleep(5)

    def download_video(self, video_url, dest_path):
        """Streams and downloads raw video file from server."""
        full_url = f"{self.server_url}{video_url}" if video_url.startswith("/") else video_url
        print(f"[*] Downloading source video from server: {full_url}")
        with requests.get(full_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] Video downloaded successfully ({round(os.path.getsize(dest_path)/(1024*1024), 2)} MB).")

    def report_progress(self, job_id, percent, current_frame, total_frames, fps, total_count, active_counts, elapsed_sec, eta_sec):
        """Sends real-time frame progress back to the Central Server."""
        url = f"{self.server_url}/api/worker/job/{job_id}/progress"
        payload = {
            "percent": int(percent),
            "current_frame": int(current_frame),
            "total_frames": int(total_frames),
            "fps": round(float(fps), 1),
            "total_count": int(total_count),
            "active_counts": active_counts,
            "elapsed_sec": round(float(elapsed_sec), 1),
            "eta_sec": round(float(eta_sec), 1)
        }
        try:
            requests.post(url, json=payload, timeout=3)
        except Exception:
            pass

    def upload_completion(self, job_id, output_video_path, events, summary):
        """Uploads completed video file and events to the Central Server."""
        url = f"{self.server_url}/api/worker/job/{job_id}/complete"
        print(f"[*] Uploading completed results for Job {job_id} to Central Server...")
        
        files = {}
        if os.path.exists(output_video_path):
            files["video"] = (os.path.basename(output_video_path), open(output_video_path, "rb"), "video/mp4")

        data = {
            "events": json.dumps(events),
            "summary": json.dumps(summary)
        }

        try:
            resp = requests.post(url, files=files, data=data, timeout=120)
            if files.get("video"):
                files["video"][1].close()
            if resp.status_code == 200:
                print(f"[OK] Job {job_id} successfully finalized on Central Server!")
                return True
            else:
                print(f"[!] Error finalizing job on server: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"[!] Failed to upload completion data: {e}")
            return False

    def process_job(self, job_data):
        """Processes a video job on local GPU using VehicleCounterSystem."""
        job_id = job_data["job_id"]
        video_url = job_data["video_url"]
        line_coords = job_data.get("line_coords", [100, 450, 1180, 450])
        conf_thresh = job_data.get("conf_thresh", 0.35)

        self.status = "busy"
        self.current_job_id = job_id
        t_start = time.time()

        print("\n" + "=" * 60)
        print(f" [*] STARTING VIDEO PROCESSING: Job {job_id}")
        print(f"     • Client GPU      : {self.gpu_info['gpu_name']}")
        print(f"     • Confidence Thresh: {conf_thresh}")
        print(f"     • Virtual Line    : {line_coords}")
        print("=" * 60)

        raw_video_path = os.path.join(STAGING_DIR, f"input_{job_id}.mp4")
        processed_video_path = os.path.join(STAGING_DIR, f"output_{job_id}.mp4")
        csv_path = os.path.join(STAGING_DIR, f"log_{job_id}.csv")

        try:
            # 1. Download source video
            self.download_video(video_url, raw_video_path)

            # 2. Initialize VehicleCounterSystem
            device = "cuda:0" if self.gpu_info["has_cuda"] else "cpu"
            system = VehicleCounterSystem(
                weights_path=self.weights_path,
                device=device,
                line_coords=tuple(line_coords),
                conf_threshold=conf_thresh,
                output_path=processed_video_path,
                save_output=True,
                csv_path=csv_path
            )

            # Progress debounce state
            last_report_time = [0.0]

            def on_progress(frame_idx, total_frames, stats):
                now = time.time()
                # Report every 15 frames or at least every 0.35 seconds
                if now - last_report_time[0] >= 0.35 or frame_idx == total_frames:
                    last_report_time[0] = now
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    elapsed = now - t_start
                    fps = frame_idx / elapsed if elapsed > 0 else 0.0
                    rem_frames = max(0, total_frames - frame_idx)
                    eta = rem_frames / fps if fps > 0 else 0.0

                    self.report_progress(
                        job_id=job_id,
                        percent=pct,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        fps=fps,
                        total_count=stats.get("total_count", 0),
                        active_counts=stats.get("class_counts", {}),
                        elapsed_sec=elapsed,
                        eta_sec=eta
                    )

            # 3. Execute Video Processing
            final_summary = system.process_video(
                video_source=raw_video_path,
                interactive=False,
                progress_callback=on_progress
            )

            # Append compute device metadata
            final_summary["device_name"] = f"{self.gpu_info['gpu_name']} ({self.worker_name})"
            final_summary["process_time"] = time.time() - t_start

            # 4. Upload completed artifacts to server
            events = system.line_counter.events if system.line_counter else []
            self.upload_completion(job_id, processed_video_path, events, final_summary)
            self.jobs_completed += 1

        except Exception as e:
            print(f"[!] Error processing job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            try:
                requests.post(
                    f"{self.server_url}/api/worker/job/{job_id}/error",
                    json={"error": str(e)},
                    timeout=5
                )
            except Exception:
                pass
        finally:
            # Cleanup temporary files
            for p in [raw_video_path, processed_video_path, csv_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.status = "idle"
            self.current_job_id = None
            print(f"[OK] Worker returned to idle state. Ready for next job!\n")

    def run(self):
        """Main worker loop."""
        self.print_banner()

        # Connect & Register
        registered = False
        while self.is_running and not registered:
            registered = self.register_with_server()
            if not registered:
                print(f"[i] Retrying connection to Central Server in 5 seconds...")
                time.sleep(5)

        # Start background heartbeat thread
        hb_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb_thread.start()

        print("[*] Worker is actively listening for processing jobs from Central Server...")
        print("[*] Press Ctrl+C in this console to safely disconnect.\n")

        poll_url = f"{self.server_url}/api/worker/job/poll?worker_id={self.worker_id}"

        while self.is_running:
            try:
                resp = requests.get(poll_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "job_available":
                        self.process_job(data)
                    else:
                        time.sleep(2.0)
                elif resp.status_code == 404:
                    # Server restarted or lost worker state, re-register
                    print("[!] Server requested re-registration. Re-registering...")
                    self.register_with_server()
                    time.sleep(2.0)
                else:
                    time.sleep(3.0)
            except requests.exceptions.RequestException:
                time.sleep(4.0)
            except Exception as e:
                print(f"[!] Error in polling loop: {e}")
                time.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(description="Indian Traffic AI - Client GPU Worker Node")
    parser.add_argument("--server", type=str, default=None, help="Central Web Server URL (e.g. http://192.168.1.111:5000)")
    parser.add_argument("--name", type=str, default=None, help="Custom friendly name for this worker node")
    args = parser.parse_args()

    cfg = load_worker_config()
    server_url = args.server or cfg.get("server_url", "http://127.0.0.1:5000")
    worker_name = args.name or cfg.get("worker_name", f"Worker-{socket.gethostname()}")

    # Prompt if server_url is still localhost or default and not specified via CLI
    if not args.server and ("127.0.0.1" in server_url or "localhost" in server_url):
        print("=" * 60)
        print("     CLIENT GPU WORKER - SERVER CONNECTION SETUP")
        print("=" * 60)
        print(f"Current Server URL: {server_url}")
        user_input = input("Enter Central Server URL (press Enter to keep current): ").strip()
        if user_input:
            server_url = user_input
            cfg["server_url"] = server_url
            save_worker_config(cfg)

    worker = ClientGPUWorker(server_url=server_url, worker_name=worker_name)
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n[*] Worker stopping by user signal. Goodbye!")
        worker.is_running = False


if __name__ == "__main__":
    main()
