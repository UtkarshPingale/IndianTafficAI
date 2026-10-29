# Indian Vehicle Detection, Classification & Virtual Line Counting System (YOLOv11)

A real-time, hardware-accelerated computer vision system for **vehicle detection, tracking, classification, and virtual line counting** on video streams, tailored specifically for Indian road traffic dynamics (including **Auto-Rickshaws, Bikes, Cars, Trucks, Buses, and Tempos/LCVs**).

Accelerated for **NVIDIA GeForce RTX 4070 SUPER** (12GB VRAM, CUDA 12.4, Tensor Cores).

---

## 🚀 Key Features

- **Virtual Line Counting**: Real-time vehicle counting when crossing a virtual tripwire on the video.
- **Direction Detection**: Automatically detects whether a vehicle is moving **Inward** or **Outward**.
- **Specialized for Indian Traffic**:
  - 🛺 **Auto-Rickshaws** (Three-wheelers)
  - 🏍️ **Bikes / Motorcycles / Scooters** (Two-wheelers)
  - 🚗 **Cars** (Hatchbacks, Sedans, SUVs, MUVs)
  - 🚚 **Trucks** (Heavy transport, lorries)
  - 🚌 **Buses & Mini-buses**
  - 🚐 **Tempos / LCVs** (Tempo-travellers, delivery vans)
- **Interactive Mouse Line Placement**: Click and drag the virtual line endpoints directly on the OpenCV video window with your mouse to adjust the line in real-time.
- **Persistent Multi-Object Tracking**: Uses YOLOv11 + ByteTrack to prevent duplicate counts and render smooth trajectory breadcrumb trails.
- **HUD Dashboard Overlay**: Real-time on-screen statistics card showing FPS, hardware status (`RTX 4070 SUPER | CUDA`), category breakdown, and total tallies.
- **Export & Reporting**: Automatically exports crossing logs to CSV (`traffic_counts.csv`) and renders annotated videos (`output_counted.mp4`).
- **GPU Accelerated Training Pipeline**: Includes `train_indian_vehicles.py` configured for RTX 4070 Super mixed-precision training.

---

## 🛠️ System Architecture

```
Vehicle-Detection-with-YOLOv11/
├── models/
│   ├── yolo11s_indian_traffic.pt     # IISc UVH-26 fine-tuned weights (14 Indian traffic classes)
│   └── uvh_classes.txt               # Class definitions
├── vehicle_counter.py                # Main inference, tracking & line counting engine
├── line_counter.py                   # Virtual line geometry, crossing intersection & deduplication
├── vehicle_classes.py                # Class mapping & color palettes
├── train_indian_vehicles.py          # Custom training & fine-tuning script for RTX 4070 Super
├── app.py                            # Streamlit web dashboard
├── traffic_counts.csv                # Generated CSV report
├── output_counted.mp4                # Generated video with annotations & HUD
└── README.md
```

---

## 📦 Standalone Windows Executable (`.exe`)

A fully self-contained Windows executable is built in `dist/IndianTrafficAI/` and runs on **any PC without requiring Python, PyTorch, or CUDA Toolkit installation**:

### 🎯 Universal Hardware GPU Acceleration
- **NVIDIA GPUs**: Automatically detects and leverages any NVIDIA GPU (GeForce RTX/GTX/Quadro) via bundled CUDA 12.4 runtime DLLs and FP16 Tensor Cores.
- **CPU Fallback**: If run on a PC without an NVIDIA GPU, it automatically falls back to optimized multi-threaded CPU mode without crashing.
- **Hardware Encoders**: Uses NVIDIA NVENC / NVDEC for video processing with automatic fallback to CPU `libx264`.

### 🚀 How to Run the `.exe`
1. **1-Click Launch**: Double-click [`Launch_IndianTrafficAI.bat`](file:///d:/Utkarsh/code/Devloper/vehical%20detection/Vehicle-Detection-with-YOLOv11/Launch_IndianTrafficAI.bat) in the root, or [`IndianTrafficAI.exe`](file:///d:/Utkarsh/code/Devloper/vehical%20detection/Vehicle-Detection-with-YOLOv11/dist/IndianTrafficAI/IndianTrafficAI.exe) inside `dist\IndianTrafficAI\`.
2. **Web Browser Studio** (Default): Launches the full web interface on `http://127.0.0.1:5000` and automatically opens your default browser for drag-and-drop video upload, interactive canvas line positioning, real-time FPS progress bar, and instant download of the rendered video and Excel report.
3. **Interactive Window Mode**:
   ```powershell
   .\dist\IndianTrafficAI\IndianTrafficAI.exe --window
   ```
4. **Batch Video Processing**:
   ```powershell
   .\dist\IndianTrafficAI\IndianTrafficAI.exe --batch --source traffic.mp4 --output counted.mp4
   ```

---

## 🔄 Automatic Remote Updates & Restart System

When you push an update, target systems running `IndianTrafficAI.exe` on other PCs will **automatically detect the update, download it, apply the new files, and restart**:

### How It Works:
1. **Remote Manifest (`version.json`)**:
   Tracks the current version, release notes, and update download link.
2. **On Target Client PCs**:
   - On startup (and in the Web Dashboard header), `IndianTrafficAI.exe` checks for new versions.
   - When a newer version is found, an **"Update Available"** alert appears (or automatic prompt).
   - The user clicks **"Update & Restart Now"** (or runs `IndianTrafficAI.exe --update`).
   - The application safely launches a detached Windows updater process, terminates the running application to release Windows file locks, copies the updated files into place, and automatically restarts `IndianTrafficAI.exe`!

### How to Publish an Update (Developer):
Run [`publish_update.bat`](file:///d:/Utkarsh/code/Devloper/vehical%20detection/Vehicle-Detection-with-YOLOv11/publish_update.bat) or CLI:
```powershell
python publish_update.py --version 1.0.1 --notes "Improved auto-rickshaw detection & new traffic features"
```
1. Automatically packages updated application files into `IndianTrafficAI_Update.zip`.
2. Updates `version.json` with the new version and changelog.
3. Push `version.json` to GitHub (or your server) and attach `IndianTrafficAI_Update.zip` to the release.
4. All client PCs automatically update and restart!

---

## ⚡ Quick Start (Python Source)

### 1. Activate Environment

```powershell
.\venv\Scripts\activate
```

### 2. Run Real-Time Counting on Video (Interactive Mode)

```powershell
python vehicle_counter.py --source compressed_output_TrafficPolice.mp4
```

**Interactive Controls in the Video Window:**
- **Mouse Drag**: Click and drag either yellow endpoint handle to reposition the virtual line across any road lane.
- `p` or `SPACE`: Pause / Resume playback
- `r`: Reset counters to zero
- `s`: Save snapshot of the current frame
- `q` or `ESC`: Exit

---

### 3. Run Headless / Batch Video Processing & Export

To process an entire video file without opening a GUI window, saving the annotated video and exporting the CSV report:

```powershell
python vehicle_counter.py `
  --source compressed_output_TrafficPolice.mp4 `
  --no-interactive `
  --save --output output_counted.mp4 `
  --save-csv --csv traffic_counts.csv
```

### 4. Custom Virtual Line Coordinates

You can pass explicit line coordinates via `--line "x1,y1,x2,y2"`:

```powershell
python vehicle_counter.py `
  --source traffic_video.mp4 `
  --line "100,500,1180,500" `
  --conf 0.35
```

---

## 🎓 Training on Custom Indian Vehicle Datasets (RTX 4070 Super)

To train or fine-tune YOLOv11 on your own dataset using CUDA acceleration and automatic mixed precision (AMP):

```powershell
python train_indian_vehicles.py `
  --data path/to/dataset.yaml `
  --model yolo11s.pt `
  --epochs 100 `
  --batch 16 `
  --imgsz 640 `
  --name indian_traffic_yolo11
```

---

## 📊 Sample CSV Report Format

The generated `traffic_counts.csv` contains detailed crossing logs:

| frame | timestamp | track_id | class | raw_class | confidence | direction | point |
|---|---|---|---|---|---|---|---|
| 6 | 15:23:53 | 2 | Car | SUV | 0.893 | Inward | (408, 440) |
| 10 | 15:23:53 | 1 | Car | SUV | 0.967 | Inward | (891, 436) |
| 41 | 15:23:54 | 25 | Bike / Two-Wheeler | Two-wheeler | 0.728 | Inward | (639, 444) |
| 136 | 15:23:57 | 47 | Car | SUV | 0.408 | Outward | (985, 426) |
| 238 | 15:24:00 | 71 | Bus | Bus | 0.444 | Inward | (643, 456) |
| 397 | 15:24:05 | 101 | Tempo / LCV | LCV | 0.465 | Inward | (852, 462) |
| 526 | 15:24:08 | 131 | Truck | Truck | 0.582 | Inward | (690, 448) |

---

## ⚙️ Hardware & Compute Performance

- **GPU**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **Inference Speed**: ~7-11ms per frame (~100-135 FPS)
- **Tracking Algorithm**: ByteTrack (with Scipy linear assignment optimization)
