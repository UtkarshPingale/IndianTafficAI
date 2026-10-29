"""
Developer Utility: Package & Publish Updates for Indian Traffic AI.
Creates an update package ZIP, updates version.json with the new version and release notes,
and guides Git commit / GitHub release upload.
"""

import os
import sys
import json
import zipfile
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(BASE_DIR, "version.json")
DIST_DIR = os.path.join(BASE_DIR, "dist", "IndianTrafficAI")


def load_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "1.0.0"}


def build_update_zip(version_str, output_zip_path):
    """Packages updated executable, scripts, and model into a deployable update archive."""
    print(f"[*] Compressing update files into: {output_zip_path}")
    
    count = 0
    single_exe = os.path.join(BASE_DIR, "dist", "IndianTrafficAI.exe")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        if os.path.exists(single_exe):
            zf.write(single_exe, "IndianTrafficAI.exe")
            count += 1
        elif os.path.exists(DIST_DIR):
            for root, dirs, files in os.walk(DIST_DIR):
                for file in files:
                    if "_update_staging" in root or "outputs" in root or "uploads" in root:
                        continue
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, DIST_DIR)
                    zf.write(full_path, rel_path)
                    count += 1

        if os.path.exists(VERSION_FILE):
            zf.write(VERSION_FILE, "version.json")
            count += 1

    print(f"[OK] Packaged {count} files into {output_zip_path} ({round(os.path.getsize(output_zip_path) / (1024**2), 1)} MB).")


def main():
    parser = argparse.ArgumentParser(description="Publish an update for Indian Traffic AI")
    parser.add_argument("--version", type=str, required=True, help="New version number (e.g. 1.0.1)")
    parser.add_argument("--notes", type=str, default="Feature updates and performance improvements.",
                        help="Release notes / changelog for this update")
    parser.add_argument("--github-user", type=str, default="UtkarshPingale", help="Your GitHub username or organization")
    parser.add_argument("--github-repo", type=str, default="IndianTafficAI", help="Your GitHub repository name")

    args = parser.parse_args()

    v_data = load_version()
    old_version = v_data.get("version", "1.0.0")
    new_version = args.version.strip().lstrip("vV")

    print("=" * 72)
    print("         INDIAN TRAFFIC AI - UPDATE PACKAGING & PUBLISHING TOOL       ")
    print("=" * 72)
    print(f"[*] Current Version : v{old_version}")
    print(f"[*] New Version     : v{new_version}")
    print(f"[*] Release Notes   : {args.notes}")
    print("-" * 72)

    # Output zip
    update_zip_name = "IndianTrafficAI_Update.zip"
    update_zip_path = os.path.join(BASE_DIR, update_zip_name)
    build_update_zip(new_version, update_zip_path)

    # Determine URLs
    if args.github_user:
        raw_manifest_url = f"https://raw.githubusercontent.com/{args.github_user}/{args.github_repo}/main/version.json"
        release_download_url = f"https://github.com/{args.github_user}/{args.github_repo}/releases/latest/download/{update_zip_name}"
    else:
        raw_manifest_url = v_data.get("update_manifest_url", "")
        release_download_url = v_data.get("download_url", "")

    # Update version.json
    v_data["version"] = new_version
    v_data["release_date"] = datetime.now().strftime("%Y-%m-%d")
    v_data["changelog"] = args.notes
    v_data["download_url"] = release_download_url
    if raw_manifest_url:
        v_data["update_manifest_url"] = raw_manifest_url

    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        json.dump(v_data, f, indent=2)

    # Also update in dist/IndianTrafficAI
    dist_ver = os.path.join(DIST_DIR, "version.json")
    with open(dist_ver, "w", encoding="utf-8") as f:
        json.dump(v_data, f, indent=2)

    print(f"[OK] Updated version.json to v{new_version}.")
    print("\n" + "=" * 72)
    print("                       NEXT STEPS TO DEPLOY UPDATE                     ")
    print("=" * 72)
    print("1. Push version.json to your GitHub repository:")
    print("   git add version.json")
    print(f"   git commit -m \"Release v{new_version}\"")
    print("   git push origin main")
    print("\n2. Attach the update ZIP to GitHub Releases:")
    print(f"   Upload '{update_zip_name}' to your GitHub Release for tag 'v{new_version}'.")
    print("\n3. That's it! Every PC running IndianTrafficAI.exe will automatically:")
    print(f"   • Detect v{new_version}")
    print(f"   • Download '{update_zip_name}'")
    print("   • Replace files and restart the system automatically!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
