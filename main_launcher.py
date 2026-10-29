"""
Universal Launcher & Entry Point for Indian Vehicle Detection AI System.
Auto-detects host system GPU (NVIDIA CUDA / Tensor Cores / CPU Fallback)
and provides unified access to:
  1. Interactive Web Dashboard Studio (Default)
  2. Live Desktop OpenCV Window with Interactive Drag-and-Drop Line
  3. Direct Batch Video File Processing & Excel Report Generation
"""

import os
import sys
import time
import argparse
import webbrowser
import threading
import torch

# Determine execution environment (frozen exe vs python script)
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR

# Ensure APP_DIR and RESOURCE_DIR are on sys.path
for p in [APP_DIR, RESOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def get_hardware_info():
    """Returns detected compute hardware details."""
    has_cuda = torch.cuda.is_available()
    device_name = "CPU (Multi-Core)"
    cuda_ver = "None"
    vram_str = "N/A"
    compute_mode = "CPU Multi-Threading"

    if has_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "Active"
            free_b, tot_b = torch.cuda.mem_get_info(0)
            free_gb = round(free_b / (1024 ** 3), 2)
            tot_gb = round(tot_b / (1024 ** 3), 2)
            vram_str = f"{tot_gb:.2f} GB ({free_gb:.2f} GB Free)"
            compute_mode = "Hardware CUDA Acceleration (FP16 Tensor Cores)"
        except Exception as e:
            has_cuda = False
            device_name = f"CPU (CUDA init error: {e})"

    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "cuda_version": cuda_ver,
        "vram": vram_str,
        "mode": compute_mode,
        "cpu_count": os.cpu_count() or 4,
        "torch_version": torch.__version__
    }


def print_banner(hw):
    print("=" * 72)
    print("      INDIAN TRAFFIC AI - VEHICLE DETECTION & VIRTUAL LINE COUNTER     ")
    print("           Powered by YOLOv11s + PyTorch CUDA Hardware Engine          ")
    print("=" * 72)
    print(f"[*] Host System Hardware Scan:")
    if hw["has_cuda"]:
        print(f"    [+] Compute Device : {hw['device_name']}")
        print(f"    [+] CUDA Engine    : {hw['cuda_version']} (Active)")
        print(f"    [+] Dedicated VRAM : {hw['vram']}")
        print(f"    [+] Acceleration   : {hw['mode']}")
    else:
        print(f"    [!] Compute Device : {hw['device_name']}")
        print(f"    [!] Acceleration   : Multi-Threaded CPU Mode ({hw['cpu_count']} Threads)")
        print(f"    [i] Notice: No NVIDIA GPU detected. Running with full CPU optimizations.")
    print("-" * 72)


def launch_web_server(port=5000, open_browser=True):
    """Launches the Flask Web Studio dashboard."""
    import web_server
    
    url = f"http://127.0.0.1:{port}"
    print(f"[*] Starting Indian Traffic AI Web Studio on {url}")
    print(f"[*] Video uploads, interactive road preview & Excel export available.")

    if open_browser:
        def _open():
            time.sleep(1.2)
            print(f"[*] Opening {url} in your default web browser...")
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print(f"[*] Server is live! Press Ctrl+C in this console window to shut down.")
    print("=" * 72 + "\n")
    web_server.app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


def launch_interactive_window(video_path=None, weights=None, line=None, conf=0.35):
    """Launches the OpenCV window with interactive mouse line dragging."""
    import vehicle_counter
    
    # If no video provided, prompt user or use file dialog
    if not video_path:
        sample_path = os.path.join(APP_DIR, "compressed_output_TrafficPolice.mp4")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(RESOURCE_DIR, "compressed_output_TrafficPolice.mp4")

        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            print("[*] Opening file picker to select a video file...")
            selected = filedialog.askopenfilename(
                title="Select Traffic Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            root.destroy()
            if selected:
                video_path = selected
            elif os.path.exists(sample_path):
                print(f"[*] No file chosen. Using bundled sample video: {sample_path}")
                video_path = sample_path
            else:
                print("[!] Error: No video file selected.")
                return
        except Exception:
            if os.path.exists(sample_path):
                video_path = sample_path
            else:
                print("[!] Error: No video source provided.")
                return

    line_coords = None
    if line:
        try:
            parts = [int(p.strip()) for p in line.split(",")]
            if len(parts) == 4:
                line_coords = tuple(parts)
        except Exception:
            pass

    weights_file = weights
    if not weights_file:
        weights_file = os.path.join(APP_DIR, "models", "yolo11s_indian_traffic.pt")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(RESOURCE_DIR, "models", "yolo11s_indian_traffic.pt")

    system = vehicle_counter.VehicleCounterSystem(
        weights_path=weights_file,
        conf_threshold=conf,
        line_coords=line_coords,
        save_output=True,
        output_path=os.path.join(APP_DIR, "outputs", "interactive_output.mp4"),
        save_csv=True,
        csv_path=os.path.join(APP_DIR, "outputs", "interactive_log.csv")
    )
    print(f"\n[*] Starting interactive OpenCV video window for: {video_path}")
    print("[*] Controls:")
    print("    • Drag yellow circular handles with mouse to adjust virtual counting line")
    print("    • Press 'p' or SPACE to Pause / Resume")
    print("    • Press 'r' to Reset vehicle counters")
    print("    • Press 's' to Save a frame snapshot")
    print("    • Press 'q' or ESC to Exit\n")
    system.process_video(video_source=video_path, interactive=True)


def launch_batch_processing(video_path, output_path=None, line=None, conf=0.35, excel=True, max_frames=None):
    """Headless high-speed video batch processor."""
    import vehicle_counter
    from excel_reporter import generate_traffic_excel
    import pandas as pd

    if not output_path:
        os.makedirs(os.path.join(APP_DIR, "outputs"), exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(APP_DIR, "outputs", f"{base}_counted.mp4")

    weights_file = os.path.join(APP_DIR, "models", "yolo11s_indian_traffic.pt")
    if not os.path.exists(weights_file):
        weights_file = os.path.join(RESOURCE_DIR, "models", "yolo11s_indian_traffic.pt")

    line_coords = None
    if line:
        try:
            parts = [int(p.strip()) for p in line.split(",")]
            if len(parts) == 4:
                line_coords = tuple(parts)
        except Exception:
            pass

    csv_path = os.path.splitext(output_path)[0] + "_log.csv"
    excel_path = os.path.splitext(output_path)[0] + "_report.xlsx"

    system = vehicle_counter.VehicleCounterSystem(
        weights_path=weights_file,
        conf_threshold=conf,
        line_coords=line_coords,
        save_output=True,
        output_path=output_path,
        save_csv=True,
        csv_path=csv_path
    )
    print(f"\n[*] Running Batch Video Processing:")
    print(f"    • Source Video : {video_path}")
    print(f"    • Output Video : {output_path}")
    print(f"    • Output CSV   : {csv_path}")

    t0 = time.time()
    system.process_video(video_source=video_path, interactive=False, max_frames=max_frames)
    elapsed = time.time() - t0

    # Build Excel
    if excel and system.line_counter:
        try:
            summary = {
                "total_count": system.line_counter.total_count,
                "inward_count": system.line_counter.direction_counts.get("Inward", 0),
                "outward_count": system.line_counter.direction_counts.get("Outward", 0),
                "class_counts": dict(system.line_counter.class_counts),
                "inward_class_counts": {},
                "outward_class_counts": {},
                "video_name": os.path.basename(video_path),
                "fps": round(system.line_counter.total_count / max(elapsed, 0.001), 1),
                "total_frames": 0,
                "device_name": system.device_name,
                "process_time": elapsed
            }
            for ev in system.line_counter.events:
                c = ev["class"]
                d = ev["direction"]
                if d == "Inward":
                    summary["inward_class_counts"][c] = summary["inward_class_counts"].get(c, 0) + 1
                elif d == "Outward":
                    summary["outward_class_counts"][c] = summary["outward_class_counts"].get(c, 0) + 1
            generate_traffic_excel(system.line_counter.events, summary, excel_path)
            print(f"    • Output Excel : {excel_path}")
        except Exception as e:
            print(f"[!] Warning: Failed to generate Excel report: {e}")

    print(f"\n[OK] Video processing completed in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Indian Traffic AI - Vehicle Detection & Counting System")
    parser.add_argument("--web", action="store_true", help="Launch Web Browser Dashboard (Default)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--window", "--interactive", dest="interactive", action="store_true",
                        help="Launch interactive OpenCV desktop window with mouse line dragging")
    parser.add_argument("--batch", action="store_true", help="Run batch headless processing without GUI window")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file for batch processing or interactive window")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--line", type=str, default=None, help="Line coordinates as 'x1,y1,x2,y2'")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--check-update", action="store_true", help="Check for available updates from remote manifest")
    parser.add_argument("--update", action="store_true", help="Download latest update, apply it, and restart system")

    args = parser.parse_args()

    hw = get_hardware_info()
    print_banner(hw)

    import auto_updater
    if args.check_update:
        print("[*] Checking for updates...")
        res = auto_updater.check_for_update()
        if res.get("update_available"):
            print(f"[!] New update available: v{res['remote_version']} (Current: v{res['current_version']})")
            print(f"    Download URL: {res.get('download_url')}")
            print(f"    Changelog: {res.get('changelog')}")
        else:
            print(f"[OK] System is up to date! (Version: {res.get('current_version')})")
        return

    if args.update:
        print("[*] Checking remote server for update package...")
        res = auto_updater.check_for_update()
        if res.get("update_available") and res.get("download_url"):
            print(f"[*] Downloading update v{res['remote_version']}...")
            auto_updater.perform_full_update_and_restart(res["download_url"])
        else:
            print(f"[OK] No newer update found. Current version: {res.get('current_version')}")
        return

    # Non-blocking background check
    threading.Thread(target=lambda: auto_updater.check_for_update(), daemon=True).start()

    def find_sample_video():
        for p in [
            os.path.join(APP_DIR, "compressed_output_TrafficPolice.mp4"),
            os.path.join(RESOURCE_DIR, "compressed_output_TrafficPolice.mp4"),
            os.path.join(os.path.dirname(APP_DIR), "compressed_output_TrafficPolice.mp4"),
            "compressed_output_TrafficPolice.mp4"
        ]:
            if os.path.exists(p):
                return p
        return "compressed_output_TrafficPolice.mp4"

    if args.batch or (args.source and args.output):
        src = args.source or find_sample_video()
        launch_batch_processing(video_path=src, output_path=args.output, line=args.line, conf=args.conf, max_frames=args.max_frames)
    elif args.interactive or (args.source and not args.batch):
        src = args.source or find_sample_video()
        launch_interactive_window(video_path=src, line=args.line, conf=args.conf)
    elif args.select:
        launch_interactive_window(video_path=None, line=args.line, conf=args.conf)
    else:
        # Default mode: Launch Web Dashboard
        launch_web_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
