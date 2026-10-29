"""
Indian Vehicle Detection, Tracking, and Virtual Line Counting System.
Optimized for NVIDIA RTX 4070 SUPER GPU with YOLOv11 and ByteTrack.

Supports:
- Auto-rickshaws, Bikes/Motorcycles, Cars, Trucks, Buses, Tempos/LCVs
- Virtual counting line with direction detection (Inward / Outward)
- Interactive mouse-draggable line in OpenCV window
- Video file processing, webcam, RTSP stream, and headless batch export
- Real-time HUD overlay with FPS and per-vehicle class statistics
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from line_counter import LineCounter
from vehicle_classes import (
    CATEGORY_COLORS,
    CATEGORY_ICONS,
    COCO_VEHICLE_CLASSES,
    UVH_CLASSES,
    get_category_color,
    map_class_name,
)


class VehicleCounterSystem:
    def __init__(
        self,
        weights_path: str = "models/yolo11s_indian_traffic.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        line_coords: Optional[Tuple[int, int, int, int]] = None,
        tracker: str = "bytetrack.yaml",
        show_trails: bool = True,
        save_output: bool = False,
        output_path: str = "output_counted.mp4",
        save_csv: bool = False,
        csv_path: str = "traffic_counts.csv",
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.show_trails = show_trails
        self.save_output = save_output
        self.output_path = output_path
        self.save_csv = save_csv
        self.csv_path = csv_path
        self.line_coords = line_coords

        # Determine runtime environment paths
        if getattr(sys, "frozen", False):
            app_dir = os.path.dirname(sys.executable)
            resource_dir = getattr(sys, "_MEIPASS", app_dir)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            resource_dir = app_dir

        # Determine compute device
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    self.device = "cuda:0"
                    self.device_name = torch.cuda.get_device_name(0)
                except Exception as e:
                    self.device = "cpu"
                    self.device_name = f"CPU (CUDA error: {e})"
            else:
                self.device = "cpu"
                self.device_name = "CPU"
        else:
            self.device = device
            self.device_name = torch.cuda.get_device_name(0) if "cuda" in device and torch.cuda.is_available() else device

        print(f"[VehicleCounterSystem] Initializing on device: {self.device} ({self.device_name})")

        # Resolve weights path
        if not os.path.exists(self.weights_path):
            candidates = [
                os.path.join(app_dir, self.weights_path),
                os.path.join(resource_dir, self.weights_path),
                os.path.join(app_dir, "models", os.path.basename(self.weights_path)),
                os.path.join(resource_dir, "models", os.path.basename(self.weights_path)),
                os.path.join(app_dir, "models", "yolo11s_indian_traffic.pt"),
                os.path.join(resource_dir, "models", "yolo11s_indian_traffic.pt"),
                "yolo11s.pt"
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.weights_path = c
                    break

        print(f"[VehicleCounterSystem] Loading YOLO model from '{self.weights_path}'...")
        self.model = YOLO(self.weights_path)
        
        # Check if the model has custom classes (e.g. UVH-26) or standard COCO
        self.model_classes = self.model.names
        self.is_uvh_model = "Three-wheeler" in list(self.model_classes.values()) or "Hatchback" in list(self.model_classes.values())

        # Target classes to detect:
        # If COCO model, filter only vehicles
        if not self.is_uvh_model:
            self.target_class_ids = [c for c in COCO_VEHICLE_CLASSES.keys() if c in self.model_classes]
        else:
            self.target_class_ids = None  # all classes in UVH-26 are vehicle/traffic classes

        # Line counter engine (initialized when first frame is read if line_coords not given)
        self.line_counter: Optional[LineCounter] = None

        # Interactive mouse dragging state
        self.dragging_pt1 = False
        self.dragging_pt2 = False
        self.hover_pt1 = False
        self.hover_pt2 = False
        self.drag_radius = 20

    def _init_line_counter(self, frame_w: int, frame_h: int):
        """Initialize virtual counting line if not specified."""
        if self.line_coords is not None:
            p1 = (self.line_coords[0], self.line_coords[1])
            p2 = (self.line_coords[2], self.line_coords[3])
        else:
            # Default: Horizontal line across the middle-lower region of frame
            y_pos = int(frame_h * 0.60)
            p1 = (int(frame_w * 0.05), y_pos)
            p2 = (int(frame_w * 0.95), y_pos)

        self.line_counter = LineCounter(p1, p2)
        print(f"[VehicleCounterSystem] Virtual Line initialized at: {p1} -> {p2}")

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback for dragging line endpoints."""
        if self.line_counter is None:
            return

        p1 = self.line_counter.pt1
        p2 = self.line_counter.pt2

        dist1 = math.hypot(x - p1[0], y - p1[1])
        dist2 = math.hypot(x - p2[0], y - p2[1])

        self.hover_pt1 = dist1 < self.drag_radius
        self.hover_pt2 = dist2 < self.drag_radius

        if event == cv2.EVENT_LBUTTONDOWN:
            if dist1 < self.drag_radius:
                self.dragging_pt1 = True
            elif dist2 < self.drag_radius:
                self.dragging_pt2 = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_pt1:
                self.line_counter.pt1 = [x, y]
            elif self.dragging_pt2:
                self.line_counter.pt2 = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_pt1 = False
            self.dragging_pt2 = False

    def draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw modern semi-transparent HUD dashboard with counts and system metrics."""
        if self.line_counter is None:
            return frame

        h, w = frame.shape[:2]

        # HUD dimensions
        card_w = 340
        card_h = 290
        margin = 15

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin, margin),
            (margin + card_w, margin + card_h),
            (20, 20, 25),
            -1
        )
        # Apply alpha blending
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Border for card
        cv2.rectangle(
            frame,
            (margin, margin),
            (margin + card_w, margin + card_h),
            (60, 60, 70),
            1
        )

        # Header Title
        title = "VEHICLE COUNTER AI"
        cv2.putText(
            frame,
            title,
            (margin + 12, margin + 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Device & FPS Badge
        fps_text = f"{fps:.1f} FPS"
        hw_text = self.device_name[:18]
        cv2.putText(
            frame,
            f"{fps_text} | {hw_text}",
            (margin + 12, margin + 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )

        # Separator line
        cv2.line(
            frame,
            (margin + 10, margin + 55),
            (margin + card_w - 10, margin + 55),
            (70, 70, 80),
            1
        )

        # Total Count Banner
        total = self.line_counter.total_count
        inward = self.line_counter.direction_counts.get("Inward", 0)
        outward = self.line_counter.direction_counts.get("Outward", 0)

        cv2.putText(
            frame,
            f"TOTAL COUNT: {total}",
            (margin + 12, margin + 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.68,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"IN: {inward}   OUT: {outward}",
            (margin + 12, margin + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Vehicle Breakdown List
        key_categories = [
            "Auto-Rickshaw",
            "Bike / Two-Wheeler",
            "Car",
            "Truck",
            "Bus",
            "Tempo / LCV",
        ]

        start_y = margin + 125
        line_spacing = 24

        for i, cat in enumerate(key_categories):
            count = self.line_counter.class_counts.get(cat, 0)
            color = get_category_color(cat)
            y_pos = start_y + i * line_spacing

            # Color dot indicator
            cv2.circle(frame, (margin + 20, y_pos - 4), 5, color, -1, cv2.LINE_AA)

            # Category Name
            cv2.putText(
                frame,
                cat,
                (margin + 34, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (230, 230, 230),
                1,
                cv2.LINE_AA
            )

            # Count
            cv2.putText(
                frame,
                str(count),
                (margin + card_w - 40, y_pos),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                color,
                1,
                cv2.LINE_AA
            )

        # Latest crossing notice at bottom of card
        if self.line_counter.last_crossed_vehicle:
            cv2.putText(
                frame,
                f"Crossed: {self.line_counter.last_crossed_vehicle}",
                (margin + 12, margin + card_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (50, 255, 100),
                1,
                cv2.LINE_AA
            )

        return frame

    def process_video(
        self,
        video_source: str,
        interactive: bool = True,
        max_frames: Optional[int] = None,
        progress_callback=None
    ) -> Dict:
        """
        Run detection, tracking, and line counting on the given video source.

        Args:
            video_source: Path to video file, webcam index ('0'), or RTSP stream.
            interactive: Whether to display OpenCV GUI window.
            max_frames: Optional limit on number of frames to process.
            progress_callback: Optional callable(frame_idx, total_frames, stats) for UI progress updates.

        Returns:
            Dictionary of final statistics.
        """
        # Open video source
        src = int(video_source) if str(video_source).isdigit() else str(video_source)
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: '{video_source}'")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VehicleCounterSystem] Video opened: {frame_w}x{frame_h} @ {fps_in:.1f} FPS ({total_frames} frames)")

        # Initialize Line Counter
        self._init_line_counter(frame_w, frame_h)

        # Video writer if output is requested
        writer = None
        if self.save_output:
            out_path = Path(self.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (frame_w, frame_h))
            print(f"[VehicleCounterSystem] Output video will be saved to: '{out_path}'")

        # Interactive window setup
        window_name = "Indian Vehicle Counter & Classifier [RTX 4070 Super]"
        if interactive:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, frame_w), min(720, frame_h))
            cv2.setMouseCallback(window_name, self.mouse_callback)

        frame_idx = 0
        t_prev = time.time()
        fps_current = fps_in
        paused = False

        try:
            while cap.isOpened():
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if max_frames and frame_idx > max_frames:
                        break

                    # YOLO tracking inference
                    # Using ByteTrack or BoT-SORT
                    track_args = {
                        "source": frame,
                        "persist": True,
                        "tracker": self.tracker,
                        "conf": self.conf_threshold,
                        "iou": self.iou_threshold,
                        "device": self.device,
                        "verbose": False
                    }
                    if self.target_class_ids is not None:
                        track_args["classes"] = self.target_class_ids

                    results = self.model.track(**track_args)[0]

                    # Process tracked detections
                    boxes = results.boxes
                    if boxes is not None and boxes.id is not None:
                        xyxy = boxes.xyxy.cpu().numpy().astype(int)
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = boxes.conf.cpu().numpy()

                        for bbox, track_id, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                            raw_name = self.model_classes.get(cls_id, "Vehicle")
                            category = map_class_name(raw_name)
                            color = get_category_color(category)

                            # Update line counter
                            self.line_counter.update_vehicle(
                                track_id=track_id,
                                bbox=bbox,
                                raw_class_name=raw_name,
                                conf=conf,
                                frame_idx=frame_idx
                            )

                            # Draw bounding box and label
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                            # Label pill
                            label = f"#{track_id} {category} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                            cv2.rectangle(
                                frame,
                                (x1, max(0, y1 - th - 6)),
                                (x1 + tw + 6, max(0, y1)),
                                color,
                                -1
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x1 + 3, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (0, 0, 0),
                                1,
                                cv2.LINE_AA
                            )

                    # Draw motion trails
                    if self.show_trails:
                        self.line_counter.draw_trajectories(frame)

                    # Draw virtual counting line
                    self.line_counter.draw_line(frame)

                    # Calculate FPS
                    t_now = time.time()
                    dt = t_now - t_prev
                    if dt > 0:
                        fps_current = 0.9 * fps_current + 0.1 * (1.0 / dt)
                    t_prev = t_now

                    # Draw HUD
                    self.draw_hud(frame, fps_current)

                    # Write frame if saving
                    if writer is not None:
                        writer.write(frame)

                    # Progress callback for web UI
                    if progress_callback is not None:
                        progress_callback(frame_idx, total_frames, self.get_summary())

                # Interactive window handling
                if interactive:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in [ord("q"), 27]:  # 'q' or ESC
                        print("[VehicleCounterSystem] Stopped by user.")
                        break
                    elif key in [ord("p"), 32]:  # 'p' or SPACE
                        paused = not paused
                        print("[VehicleCounterSystem] Paused" if paused else "[VehicleCounterSystem] Resumed")
                    elif key == ord("r"):  # 'r' to reset
                        self.line_counter.reset()
                        print("[VehicleCounterSystem] Counters reset.")
                    elif key == ord("s"):  # 's' to save screenshot
                        snap_name = f"snapshot_frame_{frame_idx}.jpg"
                        cv2.imwrite(snap_name, frame)
                        print(f"[VehicleCounterSystem] Snapshot saved to '{snap_name}'.")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if interactive:
                cv2.destroyAllWindows()

        # Save CSV report if requested
        if self.save_csv and self.line_counter and self.line_counter.events:
            df = pd.DataFrame(self.line_counter.events)
            df.to_csv(self.csv_path, index=False)
            print(f"[VehicleCounterSystem] Detailed crossing events saved to: '{self.csv_path}'")

        summary = self.get_summary()
        self.print_summary(summary)
        return summary

    def get_summary(self) -> Dict:
        """Return summary dictionary of counts and events."""
        if self.line_counter is None:
            return {}
        return {
            "total_count": self.line_counter.total_count,
            "class_counts": dict(self.line_counter.class_counts),
            "direction_counts": dict(self.line_counter.direction_counts),
            "events_count": len(self.line_counter.events)
        }

    def print_summary(self, summary: Dict):
        """Print a formatted report of results to stdout."""
        print("\n" + "=" * 55)
        print("          TRAFFIC COUNTING SUMMARY REPORT         ")
        print("=" * 55)
        print(f"Total Vehicles Counted: {summary.get('total_count', 0)}")
        print("-" * 55)
        print("Breakdown by Vehicle Category:")
        for cat, cnt in summary.get("class_counts", {}).items():
            icon = CATEGORY_ICONS.get(cat, "")
            print(f"  {icon:<10} {cat:<24}: {cnt:>5}")
        print("-" * 55)
        print("Breakdown by Direction:")
        for d, cnt in summary.get("direction_counts", {}).items():
            print(f"  {d:<24}: {cnt:>5}")
        print("=" * 55 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Indian Vehicle Detection and Virtual Line Counter")
    parser.add_argument("--source", type=str, default="compressed_output_TrafficPolice.mp4",
                        help="Path to video file, webcam index (0), or RTSP stream")
    parser.add_argument("--weights", type=str, default="models/yolo11s_indian_traffic.pt",
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--line", type=str, default=None,
                        help="Line coordinates as 'x1,y1,x2,y2'. Defaults to centered line.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda:0', 'cpu'")
    parser.add_argument("--no-interactive", action="store_true", help="Run headlessly without OpenCV window")
    parser.add_argument("--save", action="store_true", help="Save output video with annotations")
    parser.add_argument("--output", type=str, default="output_counted.mp4", help="Output video path")
    parser.add_argument("--save-csv", action="store_true", help="Save crossing event log to CSV")
    parser.add_argument("--csv", type=str, default="traffic_counts.csv", help="Output CSV path")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    return parser.parse_args()


def main():
    args = parse_args()

    line_coords = None
    if args.line:
        try:
            parts = [int(p.strip()) for p in args.line.split(",")]
            if len(parts) == 4:
                line_coords = tuple(parts)
            else:
                print(f"[Warning] Invalid line format '{args.line}'. Expected 'x1,y1,x2,y2'. Using auto line.")
        except Exception as e:
            print(f"[Warning] Failed to parse line coordinates: {e}. Using auto line.")

    system = VehicleCounterSystem(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        line_coords=line_coords,
        save_output=args.save,
        output_path=args.output,
        save_csv=args.save_csv,
        csv_path=args.csv
    )

    system.process_video(
        video_source=args.source,
        interactive=not args.no_interactive,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
