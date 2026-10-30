"""
Universal Auto-Updater for Indian Traffic AI (.exe & Source).
Handles:
  1. Remote version checking (GitHub Releases, raw manifest, or custom HTTP URL)
  2. Background update package download with progress streaming
  3. Safe Windows file replacement (handles locked .exe using detached updater batch script)
  4. Automatic application restart
"""

import os
import sys
import json
import time
import zipfile
import shutil
import base64
import urllib.request
import subprocess
from pathlib import Path

# Runtime directory resolution
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
    EXE_NAME = os.path.basename(sys.executable)
    IS_FROZEN = True
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR
    EXE_NAME = "main_launcher.py"
    IS_FROZEN = False

DEFAULT_VERSION_FILE = os.path.join(APP_DIR, "version.json")
CONFIG_FILE = os.path.join(APP_DIR, "update_config.json")
UPDATE_STAGING_DIR = os.path.join(APP_DIR, "_update_staging")

DEFAULT_CONFIG = {
    "version": "1.0.0",
    "app_name": "Indian Traffic AI",
    "release_date": "2026-10-29",
    # Default manifest URL: points to UtkarshPingale/IndianTafficAI
    "update_manifest_url": "https://raw.githubusercontent.com/UtkarshPingale/IndianTafficAI/main/version.json",
    "download_url": "https://github.com/UtkarshPingale/IndianTafficAI/releases/latest/download/IndianTrafficAI_Update.zip",
    "changelog": "Initial hardware-accelerated release with YOLOv11s and PyTorch CUDA.",
    "auto_check_on_launch": True,
    "auto_restart_after_update": True
}


def load_config() -> dict:
    """Loads configuration and version details, falling back to defaults."""
    cfg = DEFAULT_CONFIG.copy()
    
    # Check config file first, then version file
    for filepath in [CONFIG_FILE, DEFAULT_VERSION_FILE, os.path.join(RESOURCE_DIR, "version.json")]:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cfg.update(data)
                break
            except Exception:
                pass
    return cfg


def save_config(cfg: dict) -> bool:
    """Saves updated configuration to file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"[AutoUpdater] Failed to save config: {e}")
        return False


def parse_version_tuple(v_str: str) -> tuple:
    """Parses a version string like '1.2.3' into a comparable tuple (1, 2, 3)."""
    try:
        # Strip leading 'v' if present (e.g. 'v1.0.1' -> '1.0.1')
        v_clean = str(v_str).strip().lstrip("vV")
        parts = [int(p) for p in v_clean.split(".") if p.isdigit()]
        return tuple(parts) if parts else (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def check_for_update(manifest_url: str = None, timeout: int = 5) -> dict:
    """
    Checks remote manifest URL to see if a newer version is available.
    
    Returns:
        dict: {
            "status": "ok" / "error",
            "current_version": str,
            "remote_version": str,
            "update_available": bool,
            "download_url": str,
            "changelog": str,
            "release_date": str,
            "error": str or None
        }
    """
    cfg = load_config()
    current_ver = cfg.get("version", "1.0.0")
    url = manifest_url or cfg.get("update_manifest_url")

    result = {
        "status": "ok",
        "current_version": current_ver,
        "remote_version": current_ver,
        "update_available": False,
        "download_url": "",
        "changelog": "",
        "release_date": "",
        "error": None
    }

    if not url or url.startswith("http://example") or "YOUR_REPO" in url:
        result["status"] = "unconfigured"
        result["error"] = "Update URL not configured."
        return result

    try:
        # Add cache buster query parameter to bypass CDN/proxy caching
        cache_buster = f"?t={int(time.time())}"
        fetch_url = url + cache_buster if "?" not in url else url + f"&t={int(time.time())}"
        
        req = urllib.request.Request(
            fetch_url,
            headers={"User-Agent": "IndianTrafficAI-Updater/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                remote_data = json.loads(response.read().decode("utf-8"))
                remote_ver = remote_data.get("version", current_ver)
                
                curr_tuple = parse_version_tuple(current_ver)
                remote_tuple = parse_version_tuple(remote_ver)

                result["remote_version"] = remote_ver
                result["download_url"] = remote_data.get("download_url", "")
                result["changelog"] = remote_data.get("changelog", "No release notes provided.")
                result["release_date"] = remote_data.get("release_date", "")
                result["update_available"] = remote_tuple > curr_tuple

    except urllib.error.URLError as e:
        result["status"] = "error"
        result["error"] = f"Network error checking update: {e.reason}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Failed to check for updates: {str(e)}"

    return result


def download_update(download_url: str, progress_callback=None) -> str:
    """
    Downloads update archive into staging directory.
    
    Args:
        download_url: Direct download link to update ZIP archive.
        progress_callback: Optional callable(percent, downloaded_bytes, total_bytes).
        
    Returns:
        str: Absolute path to the downloaded ZIP file.
    """
    if not download_url:
        raise ValueError("Download URL cannot be empty.")

    os.makedirs(UPDATE_STAGING_DIR, exist_ok=True)
    zip_dest = os.path.join(UPDATE_STAGING_DIR, "update_package.zip")

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "IndianTrafficAI-Updater/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 65536  # 64 KB chunks

        with open(zip_dest, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    pct = int((downloaded / total_size) * 100) if total_size > 0 else 0
                    progress_callback(pct, downloaded, total_size)

    return zip_dest


def extract_update_package(zip_path: str) -> str:
    """Extracts update archive into an extraction staging folder."""
    extracted_dir = os.path.join(UPDATE_STAGING_DIR, "extracted")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir, ignore_errors=True)
    os.makedirs(extracted_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)

    return extracted_dir


def launch_detached_updater_and_restart(extracted_dir: str):
    """
    Spawns an in-memory, completely detached background updater via Windows PowerShell.
    NO apply_update.bat or batch script file is created on disk.
    Waits for this process to terminate, replaces files seamlessly, cleans up staging,
    and restarts Indian Traffic AI automatically.
    """
    pid = os.getpid()
    target_dir = os.path.abspath(APP_DIR)
    source_dir = os.path.abspath(extracted_dir)
    staging_dir = os.path.abspath(UPDATE_STAGING_DIR)

    if IS_FROZEN:
        exe_path = os.path.abspath(os.path.join(APP_DIR, EXE_NAME))
        restart_stmt = f'Start-Process -FilePath "{exe_path}" -WorkingDirectory "{target_dir}"'
    else:
        exe_path = os.path.abspath(sys.executable)
        launcher_py = os.path.abspath(os.path.join(APP_DIR, "main_launcher.py"))
        restart_stmt = f'Start-Process -FilePath "{exe_path}" -ArgumentList "{launcher_py}" -WorkingDirectory "{target_dir}"'

    ps_script = f"""
$procId = {pid}
$source = '{source_dir}'
$dest = '{target_dir}'
$staging = '{staging_dir}'

# 1. Wait for current process to exit and release file locks
if ($procId -gt 0) {{
    try {{
        Wait-Process -Id $procId -Timeout 15 -ErrorAction SilentlyContinue
    }} catch {{}}
}}
Start-Sleep -Milliseconds 700

# 2. Overwrite files from update package with retry loop
for ($i = 0; $i -lt 10; $i++) {{
    try {{
        Copy-Item -Path "$source\\*" -Destination "$dest" -Recurse -Force -ErrorAction Stop
        break
    }} catch {{
        Start-Sleep -Milliseconds 500
    }}
}}

# 3. Clean up staging cache and any leftover old bat files
try {{
    Remove-Item -Path "$staging" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $dest "apply_update.bat") -Force -ErrorAction SilentlyContinue
}} catch {{}}

# 4. Seamlessly relaunch the application
try {{
    {restart_stmt}
}} catch {{}}
"""

    encoded_cmd = base64.b64encode(ps_script.encode("utf-16le")).decode("ascii")
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-WindowStyle",
        "Hidden",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded_cmd
    ]

    print("[*] Launching seamless in-memory background updater (no .bat files created)...")
    print("[*] Terminating current process to release locks and restart automatically...")

    DETACHED_FLAGS = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        cmd,
        creationflags=DETACHED_FLAGS,
        close_fds=True
    )

    time.sleep(0.5)
    sys.exit(0)


def generate_updater_batch(target_dir: str, source_dir: str, exe_path: str, pid: int) -> str:
    """Deprecated: in-memory updater is used instead of creating .bat files."""
    return ""


def apply_and_restart(extracted_dir: str):
    """Seamlessly updates application files and restarts without creating any .bat files."""
    launch_detached_updater_and_restart(extracted_dir)


def perform_full_update_and_restart(download_url: str, progress_callback=None):
    """
    One-call orchestrator: downloads update, extracts files, launches updater script,
    and restarts the application.
    """
    print(f"[*] Starting update download from: {download_url}")
    zip_path = download_update(download_url, progress_callback=progress_callback)
    print(f"[*] Extracting update package...")
    extracted_dir = extract_update_package(zip_path)
    print(f"[*] Applying update and restarting system...")
    apply_and_restart(extracted_dir)
