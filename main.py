"""
Indian Traffic AI - Vehicle Detection, Tracking & Virtual Line Counter
All-In-One Unified System (Standalone Engine, Web Studio & Interactive OpenCV GUI)

Features:
- YOLOv11s Indian Traffic Detection (Auto-rickshaws, Bikes, Cars, Buses, Trucks, Tempos)
- ByteTrack & BoT-SORT Object Tracking
- Virtual Counting Line with Direction Detection (Inward / Outward)
- Interactive Mouse Drag-and-Drop Line in Desktop OpenCV Window
- Complete Flask Web Studio with Live Road Preview & Real-Time Analytics
- Styled Multi-Sheet Excel Reports & CSV Event Auditing
- Automatic Hardware Detection (NVIDIA CUDA / Tensor Cores / CPU Fallback)
"""

import os
import sys
import math
import time
import json
import uuid
import socket
import argparse
import threading
import subprocess
import webbrowser
from datetime import datetime
from collections import defaultdict, deque, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Reconfigure standard streams for Windows terminal Unicode compatibility
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask.json.provider import DefaultJSONProvider

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# =============================================================================
# ENVIRONMENT & PATH CONFIGURATION
# =============================================================================

if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = APP_DIR

for p in [APP_DIR, RESOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

STATIC_DIR = os.path.join(RESOURCE_DIR, "static")
TEMPLATES_DIR = os.path.join(RESOURCE_DIR, "templates")
UPLOADS_DIR = os.path.join(APP_DIR, "uploads")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")
MODELS_DIR = os.path.join(APP_DIR, "models")

if not os.path.exists(os.path.join(MODELS_DIR, "yolo11s_indian_traffic.pt")):
    MODELS_DIR = os.path.join(RESOURCE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolo11s_indian_traffic.pt")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

SAMPLE_VIDEO = os.path.join(APP_DIR, "compressed_output_TrafficPolice.mp4")
if not os.path.exists(SAMPLE_VIDEO):
    SAMPLE_VIDEO = os.path.join(RESOURCE_DIR, "compressed_output_TrafficPolice.mp4")


# =============================================================================
# VEHICLE CLASSES & COLOR PALETTES
# =============================================================================

UVH_CLASSES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck",
    "Three-wheeler", "Two-wheeler", "LCV", "Mini-bus",
    "Tempo-traveller", "Bicycle", "Van", "Others"
]

UVH_TO_CATEGORY: Dict[str, str] = {
    "Three-wheeler": "Auto-Rickshaw",
    "Two-wheeler": "Bike / Two-Wheeler",
    "Hatchback": "Car",
    "Sedan": "Car",
    "SUV": "Car",
    "MUV": "Car",
    "Truck": "Truck",
    "Bus": "Bus",
    "Mini-bus": "Bus",
    "LCV": "Tempo / LCV",
    "Tempo-traveller": "Tempo / LCV",
    "Van": "Tempo / LCV",
    "Bicycle": "Bicycle",
    "Others": "Other Vehicle"
}

COCO_VEHICLE_CLASSES: Dict[int, str] = {
    1: "Bicycle",
    2: "Car",
    3: "Bike / Two-Wheeler",
    5: "Bus",
    7: "Truck"
}

CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Auto-Rickshaw": (0, 215, 255),       # Gold / Amber Yellow
    "Bike / Two-Wheeler": (0, 140, 255),  # Deep Orange
    "Car": (255, 191, 0),                 # Deep Sky Blue
    "Truck": (50, 50, 255),               # Bright Red
    "Bus": (255, 0, 255),                 # Magenta
    "Tempo / LCV": (0, 255, 127),         # Spring Green
    "Bicycle": (205, 235, 139),           # Pale Green
    "Other Vehicle": (180, 180, 180),     # Gray
    "Vehicle": (0, 255, 255)              # Yellow fallback
}

CATEGORY_ICONS: Dict[str, str] = {
    "Auto-Rickshaw": "[AUTO]",
    "Bike / Two-Wheeler": "[BIKE]",
    "Car": "[CAR]",
    "Truck": "[TRUCK]",
    "Bus": "[BUS]",
    "Tempo / LCV": "[TEMPO]",
    "Bicycle": "[CYCLE]",
    "Other Vehicle": "[OTHER]",
    "Vehicle": "[VEHICLE]"
}


def map_class_name(raw_name: str) -> str:
    """Map a raw detection label to an aggregated Indian vehicle category."""
    if raw_name in UVH_TO_CATEGORY:
        return UVH_TO_CATEGORY[raw_name]
    
    lower = raw_name.lower()
    if any(k in lower for k in ["three", "auto", "rickshaw", "cng", "toto"]):
        return "Auto-Rickshaw"
    if any(k in lower for k in ["two", "motorcycle", "bike", "scooter", "moped"]):
        return "Bike / Two-Wheeler"
    if any(k in lower for k in ["car", "hatchback", "sedan", "suv", "muv", "jeep"]):
        return "Car"
    if any(k in lower for k in ["truck", "lorry", "trailer"]):
        return "Truck"
    if any(k in lower for k in ["bus", "mini-bus", "minibus"]):
        return "Bus"
    if any(k in lower for k in ["lcv", "tempo", "van", "pickup"]):
        return "Tempo / LCV"
    if any(k in lower for k in ["bicycle", "cycle"]):
        return "Bicycle"
    
    return raw_name


def get_category_color(category: str) -> Tuple[int, int, int]:
    """Return BGR color tuple for a category."""
    return CATEGORY_COLORS.get(category, (0, 255, 0))


# =============================================================================
# VIRTUAL LINE COUNTER & GEOMETRY
# =============================================================================

def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Return True if points A, B, C are in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float]
) -> bool:
    """Check if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class LineCounter:
    """Tracks vehicle centroids across a virtual line and counts them per class and direction."""

    def __init__(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        line_color: Tuple[int, int, int] = (0, 255, 255),
        flash_color: Tuple[int, int, int] = (0, 255, 0),
        line_thickness: int = 3,
        trail_length: int = 25
    ):
        self.pt1 = list(pt1)
        self.pt2 = list(pt2)
        self.line_color = line_color
        self.flash_color = flash_color
        self.line_thickness = line_thickness
        self.trail_length = trail_length

        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_length))
        self.track_classes: Dict[int, str] = {}
        self.track_confs: Dict[int, float] = {}
        self.counted_ids: Set[int] = set()

        self.class_counts: Counter = Counter()
        self.direction_counts: Counter = Counter({"Inward": 0, "Outward": 0})
        self.total_count: int = 0
        self.events: List[Dict] = []

        self.flash_frames = 0
        self.last_crossed_vehicle = ""

    def set_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Update line coordinates."""
        self.pt1 = list(pt1)
        self.pt2 = list(pt2)

    def side_of_line(self, point: Tuple[float, float]) -> float:
        """Signed 2D cross product indicates which side of the line the point is on."""
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def update_vehicle(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        raw_class_name: str,
        conf: float,
        frame_idx: int = 0
    ) -> Optional[Dict]:
        category = map_class_name(raw_class_name)
        self.track_classes[track_id] = category
        self.track_confs[track_id] = max(self.track_confs.get(track_id, 0.0), conf)

        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        curr_pt = (cx, cy)

        history = self.track_history[track_id]
        event = None

        if len(history) > 0 and track_id not in self.counted_ids:
            prev_pt = history[-1]
            L1 = (float(self.pt1[0]), float(self.pt1[1]))
            L2 = (float(self.pt2[0]), float(self.pt2[1]))
            P1 = (float(prev_pt[0]), float(prev_pt[1]))
            P2 = (float(curr_pt[0]), float(curr_pt[1]))

            if segments_intersect(L1, L2, P1, P2):
                side_prev = self.side_of_line(prev_pt)
                direction = "Inward" if side_prev < 0 else "Outward"

                self.counted_ids.add(track_id)
                self.class_counts[category] += 1
                self.direction_counts[direction] += 1
                self.total_count += 1
                self.flash_frames = 12
                self.last_crossed_vehicle = f"{category} (#{track_id})"

                event = {
                    "frame": int(frame_idx),
                    "timestamp": time.strftime("%H:%M:%S"),
                    "track_id": int(track_id),
                    "class": str(category),
                    "raw_class": str(raw_class_name),
                    "confidence": float(round(float(conf), 3)),
                    "direction": str(direction),
                    "point": f"({int(curr_pt[0])}, {int(curr_pt[1])})"
                }
                self.events.append(event)

        history.append(curr_pt)
        return event

    def draw_line(self, frame: np.ndarray) -> np.ndarray:
        p1 = (int(self.pt1[0]), int(self.pt1[1]))
        p2 = (int(self.pt2[0]), int(self.pt2[1]))

        if self.flash_frames > 0:
            color = self.flash_color
            thick = self.line_thickness + 3
            self.flash_frames -= 1
        else:
            color = self.line_color
            thick = self.line_thickness

        cv2.line(frame, p1, p2, (0, 100, 100), thick + 4, cv2.LINE_AA)
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)

        for pt in [p1, p2]:
            cv2.circle(frame, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            frame,
            "VIRTUAL COUNTING LINE",
            (mid_x - 90, max(25, mid_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )
        return frame

    def draw_trajectories(self, frame: np.ndarray, active_ids: Optional[Set[int]] = None) -> np.ndarray:
        targets = [tid for tid in self.track_history if (active_ids is None or tid in active_ids)]
        for track_id in targets:
            pts = self.track_history[track_id]
            if len(pts) < 2:
                continue
            cat = self.track_classes.get(track_id, "Vehicle")
            color = get_category_color(cat)

            num_pts = len(pts)
            for i in range(1, num_pts):
                thickness = int(math.sqrt(64 / float(num_pts - i + 1)))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        if len(self.track_history) > 120 and active_ids is not None:
            stale_keys = [k for k in self.track_history.keys() if k not in active_ids and k in self.counted_ids]
            for k in stale_keys:
                del self.track_history[k]

        return frame

    def reset(self):
        self.track_history.clear()
        self.track_classes.clear()
        self.track_confs.clear()
        self.counted_ids.clear()
        self.class_counts.clear()
        self.direction_counts = Counter({"Inward": 0, "Outward": 0})
        self.total_count = 0
        self.events.clear()
        self.flash_frames = 0
        self.last_crossed_vehicle = ""


# =============================================================================
# EXCEL AUDIT REPORT GENERATOR
# =============================================================================

def generate_traffic_excel(events_data, summary_data, output_excel_path):
    """Generates a styled multi-sheet Excel file (.xlsx) with summary KPIs and crossing event logs."""
    wb = openpyxl.Workbook()
    
    font_title = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    font_subtitle = Font(name="Calibri", size=10, italic=True, color="94A3B8")
    font_section = Font(name="Calibri", size=12, bold=True, color="0F172A")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_data = Font(name="Calibri", size=11, color="1E293B")
    font_bold = Font(name="Calibri", size=11, bold=True, color="1E293B")
    
    fill_dark_navy = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    fill_header_blue = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    fill_header_teal = PatternFill(start_color="0E7490", end_color="0E7490", fill_type="solid")
    fill_zebra_light = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    fill_highlight_green = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_highlight_yellow = PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid")
    
    border_thin = Side(border_style="thin", color="CBD5E1")
    cell_border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)
    border_top_double = Border(top=Side(border_style="double", color="0F172A"), bottom=Side(border_style="thin", color="0F172A"))
    
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")

    # SHEET 1: TRAFFIC SUMMARY & METRICS
    ws_sum = wb.active
    ws_sum.title = "Traffic Summary"
    ws_sum.views.sheetView[0].showGridLines = True
    
    ws_sum.merge_cells("A1:F1")
    title_cell = ws_sum["A1"]
    title_cell.value = "INDIAN VEHICLE DETECTION & TRAFFIC AUDIT REPORT"
    title_cell.font = font_title
    title_cell.fill = fill_dark_navy
    title_cell.alignment = align_center
    ws_sum.row_dimensions[1].height = 36
    
    ws_sum.merge_cells("A2:F2")
    sub_cell = ws_sum["A2"]
    sub_cell.value = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine: YOLOv11 Indian Traffic Model | GPU: {summary_data.get('device_name', 'NVIDIA GPU')}"
    sub_cell.font = font_subtitle
    sub_cell.fill = fill_dark_navy
    sub_cell.alignment = align_center
    ws_sum.row_dimensions[2].height = 20

    ws_sum.cell(row=4, column=1, value="VIDEO SPECIFICATIONS").font = font_section
    info_rows = [
        ("Source Video", summary_data.get("video_name", "N/A")),
        ("Total Frames Processed", summary_data.get("total_frames", 0)),
        ("Processing Speed", f"{summary_data.get('fps', 0):.1f} FPS"),
        ("Compute Hardware", summary_data.get("device_name", "NVIDIA GPU")),
        ("Audit Generation Time", f"{summary_data.get('process_time', 0):.2f} seconds"),
    ]
    
    r = 5
    for label, val in info_rows:
        c1 = ws_sum.cell(row=r, column=1, value=label)
        c2 = ws_sum.cell(row=r, column=2, value=val)
        c1.font = font_bold
        c2.font = font_data
        c1.fill = fill_zebra_light
        c1.border = cell_border
        c2.border = cell_border
        ws_sum.row_dimensions[r].height = 20
        r += 1
        
    ws_sum.cell(row=4, column=4, value="TOTAL TRAFFIC FLOW METRICS").font = font_section
    kpi_rows = [
        ("Total Vehicles Counted", summary_data.get("total_count", 0)),
        ("Inward Flow (Towards)", summary_data.get("inward_count", 0)),
        ("Outward Flow (Away)", summary_data.get("outward_count", 0)),
    ]
    r_kpi = 5
    for label, val in kpi_rows:
        c1 = ws_sum.cell(row=r_kpi, column=4, value=label)
        c2 = ws_sum.cell(row=r_kpi, column=5, value=val)
        c1.font = font_bold
        c2.font = Font(name="Calibri", size=12, bold=True, color="0E7490" if "Inward" in label else "059669")
        c1.fill = fill_zebra_light
        c1.border = cell_border
        c2.border = cell_border
        c2.alignment = align_center
        ws_sum.row_dimensions[r_kpi].height = 20
        r_kpi += 1

    r_start = 12
    ws_sum.cell(row=r_start, column=1, value="INDIAN VEHICLE CLASSIFICATION BREAKDOWN").font = font_section
    
    headers_cls = ["Vehicle Class", "Total Count", "Inward Count", "Outward Count", "Traffic Share (%)"]
    r_head = r_start + 1
    ws_sum.row_dimensions[r_head].height = 26
    for col_idx, h in enumerate(headers_cls, start=1):
        c = ws_sum.cell(row=r_head, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header_blue
        c.alignment = align_center
        c.border = cell_border

    class_counts = summary_data.get("class_counts", {})
    inward_counts = summary_data.get("inward_class_counts", {})
    outward_counts = summary_data.get("outward_class_counts", {})
    total_all = summary_data.get("total_count", 0)

    all_classes = [
        "Auto-Rickshaw", "Bike / Two-Wheeler", "Car", "Bus", "Truck",
        "Tempo / LCV", "Bicycle", "Other Vehicle"
    ]
    for k in class_counts.keys():
        if k not in all_classes:
            all_classes.append(k)

    curr_row = r_head + 1
    for cls_name in all_classes:
        t_cnt = class_counts.get(cls_name, 0)
        i_cnt = inward_counts.get(cls_name, 0)
        o_cnt = outward_counts.get(cls_name, 0)
        pct = (t_cnt / total_all * 100) if total_all > 0 else 0.0

        c_cls = ws_sum.cell(row=curr_row, column=1, value=cls_name)
        c_tot = ws_sum.cell(row=curr_row, column=2, value=t_cnt)
        c_in = ws_sum.cell(row=curr_row, column=3, value=i_cnt)
        c_out = ws_sum.cell(row=curr_row, column=4, value=o_cnt)
        c_pct = ws_sum.cell(row=curr_row, column=5, value=f"{pct:.1f}%")

        c_cls.font = font_bold if t_cnt > 0 else font_data
        c_tot.font = font_data
        c_in.font = font_data
        c_out.font = font_data
        c_pct.font = font_data

        c_tot.alignment = align_center
        c_in.alignment = align_center
        c_out.alignment = align_center
        c_pct.alignment = align_center

        for cell in (c_cls, c_tot, c_in, c_out, c_pct):
            cell.border = cell_border
            if curr_row % 2 == 0:
                cell.fill = fill_zebra_light

        ws_sum.row_dimensions[curr_row].height = 20
        curr_row += 1

    ws_sum.row_dimensions[curr_row].height = 24
    c_tot_lbl = ws_sum.cell(row=curr_row, column=1, value="GRAND TOTAL")
    c_tot_cnt = ws_sum.cell(row=curr_row, column=2, value=total_all)
    c_tot_in = ws_sum.cell(row=curr_row, column=3, value=summary_data.get("inward_count", 0))
    c_tot_out = ws_sum.cell(row=curr_row, column=4, value=summary_data.get("outward_count", 0))
    c_tot_pct = ws_sum.cell(row=curr_row, column=5, value="100.0%")

    for cell in (c_tot_lbl, c_tot_cnt, c_tot_in, c_tot_out, c_tot_pct):
        cell.font = font_bold
        cell.border = border_top_double
        cell.fill = fill_highlight_yellow
        cell.alignment = align_center
    c_tot_lbl.alignment = align_left

    for col in ws_sum.iter_cols(min_row=4, max_col=6):
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len and not cell.coordinate in ("A1", "A2"):
                max_len = len(val_str)
        ws_sum.column_dimensions[col_letter].width = max(max_len + 4, 18)

    # SHEET 2: DETAILED CROSSING EVENTS LOG
    ws_events = wb.create_sheet(title="Crossing Events Log")
    ws_events.views.sheetView[0].showGridLines = True

    ws_events.merge_cells("A1:I1")
    ev_title = ws_events["A1"]
    ev_title.value = "VEHICLE CROSSING AUDIT LOG (PER-VEHICLE RECORDS)"
    ev_title.font = font_title
    ev_title.fill = fill_dark_navy
    ev_title.alignment = align_center
    ws_events.row_dimensions[1].height = 32

    event_headers = [
        "S.No", "Frame Number", "Timestamp (s)", "Track ID",
        "Vehicle Category", "Raw Sub-Class", "Confidence (%)",
        "Flow Direction", "Crossing Point (X, Y)"
    ]

    ws_events.row_dimensions[3].height = 26
    for col_idx, h in enumerate(event_headers, start=1):
        c = ws_events.cell(row=3, column=col_idx, value=h)
        c.font = font_header
        c.fill = fill_header_teal
        c.alignment = align_center
        c.border = cell_border

    row_idx = 4
    for idx, ev in enumerate(events_data, start=1):
        conf_val = float(ev.get("confidence", 0.0))
        conf_str = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.1f}%"

        c_sno = ws_events.cell(row=row_idx, column=1, value=idx)
        c_frm = ws_events.cell(row=row_idx, column=2, value=ev.get("frame", 0))
        c_time = ws_events.cell(row=row_idx, column=3, value=ev.get("timestamp", ""))
        c_tid = ws_events.cell(row=row_idx, column=4, value=ev.get("track_id", 0))
        c_cls = ws_events.cell(row=row_idx, column=5, value=ev.get("class", ""))
        c_raw = ws_events.cell(row=row_idx, column=6, value=ev.get("raw_class", ""))
        c_conf = ws_events.cell(row=row_idx, column=7, value=conf_str)
        c_dir = ws_events.cell(row=row_idx, column=8, value=ev.get("direction", ""))
        c_pt = ws_events.cell(row=row_idx, column=9, value=str(ev.get("point", "")))

        for cell in (c_sno, c_frm, c_time, c_tid, c_cls, c_raw, c_conf, c_dir, c_pt):
            cell.font = font_data
            cell.border = cell_border
            cell.alignment = align_center
            if row_idx % 2 == 0:
                cell.fill = fill_zebra_light

        if ev.get("direction") == "Inward":
            c_dir.fill = fill_highlight_green
            c_dir.font = Font(name="Calibri", size=11, bold=True, color="047857")
        elif ev.get("direction") == "Outward":
            c_dir.fill = fill_highlight_yellow
            c_dir.font = Font(name="Calibri", size=11, bold=True, color="B45309")

        ws_events.row_dimensions[row_idx].height = 20
        row_idx += 1

    for col in ws_events.iter_cols(min_row=3, max_col=9):
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len and cell.coordinate != "A1":
                max_len = len(val_str)
        ws_events.column_dimensions[col_letter].width = max(max_len + 4, 14)

    os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)
    wb.save(output_excel_path)
    return output_excel_path


# =============================================================================
# CORE VEHICLE COUNTER SYSTEM (OPENCV & BATCH ENGINE)
# =============================================================================

class VehicleCounterSystem:
    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
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

        # Compute device configuration
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    self.device = "cuda:0"
                    self.device_name = torch.cuda.get_device_name(0)
                    t_test = torch.zeros((1, 1), device="cuda:0") + 1.0
                    del t_test
                    self.use_fp16 = True
                except Exception:
                    self.device = "cpu"
                    self.device_name = "CPU (Multi-Threaded Fallback)"
                    self.use_fp16 = False
            else:
                self.device = "cpu"
                self.device_name = "CPU (Multi-Threaded)"
                self.use_fp16 = False
        else:
            self.device = device
            self.device_name = device
            self.use_fp16 = "cuda" in device and torch.cuda.is_available()

        print(f"[*] Initializing YOLOv11 on compute hardware: {self.device_name}")
        self.model = YOLO(self.weights_path)

        if self.device.startswith("cuda"):
            try:
                self.model.to(self.device)
                dummy = np.zeros((384, 640, 3), dtype=np.uint8)
                if self.use_fp16:
                    try:
                        self.model.model.half()
                        self.model.predict(dummy, device=self.device, verbose=False, imgsz=640)
                    except Exception:
                        self.model.model.float()
                        self.use_fp16 = False
                        self.model.predict(dummy, device=self.device, verbose=False, imgsz=640)
                else:
                    self.model.model.float()
                    self.model.predict(dummy, device=self.device, verbose=False, imgsz=640)
            except Exception as e:
                print(f"[!] Warning: CUDA execution failed on this GPU ({e}). Falling back to CPU...")
                self.device = "cpu"
                self.device_name = "CPU Mode (Fallback)"
                self.use_fp16 = False
                self.model.to("cpu")
                self.model.model.float()

        self.line_counter: Optional[LineCounter] = None
        self.dragging_point: Optional[int] = None
        self.handle_radius: int = 15
        self.total_frames_processed = 0
        self.start_time = 0.0

    def _init_line_counter(self, frame_w: int, frame_h: int):
        if self.line_coords is not None:
            pt1 = (self.line_coords[0], self.line_coords[1])
            pt2 = (self.line_coords[2], self.line_coords[3])
        else:
            y = int(frame_h * 0.60)
            pt1 = (int(frame_w * 0.05), y)
            pt2 = (int(frame_w * 0.95), y)
        self.line_counter = LineCounter(pt1=pt1, pt2=pt2)

    def mouse_callback(self, event, x, y, flags, param):
        if self.line_counter is None:
            return

        p1 = self.line_counter.pt1
        p2 = self.line_counter.pt2

        if event == cv2.EVENT_LBUTTONDOWN:
            d1 = math.hypot(x - p1[0], y - p1[1])
            d2 = math.hypot(x - p2[0], y - p2[1])
            if d1 <= self.handle_radius:
                self.dragging_point = 1
            elif d2 <= self.handle_radius:
                self.dragging_point = 2

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point == 1:
                self.line_counter.set_line((x, y), tuple(p2))
            elif self.dragging_point == 2:
                self.line_counter.set_line(tuple(p1), (x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

    def draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        if self.line_counter is None:
            return frame

        fh, fw = frame.shape[:2]
        hud_w = 340
        hud_h = 240
        hud_x = fw - hud_w - 15
        hud_y = 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 191, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "TRAFFIC ANALYTICS", (hud_x + 12, hud_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        fps_color = (0, 255, 0) if fps >= 30 else (0, 215, 255)
        cv2.putText(frame, f"{fps:.1f} FPS", (hud_x + hud_w - 75, hud_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"TOTAL COUNT: {self.line_counter.total_count}",
                    (hud_x + 12, hud_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        in_cnt = self.line_counter.direction_counts.get("Inward", 0)
        out_cnt = self.line_counter.direction_counts.get("Outward", 0)
        cv2.putText(frame, f"IN: {in_cnt}", (hud_x + 12, hud_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 127), 2, cv2.LINE_AA)
        cv2.putText(frame, f"OUT: {out_cnt}", (hud_x + 120, hud_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255), 2, cv2.LINE_AA)

        y_offset = hud_y + 102
        top_classes = [
            ("Auto-Rickshaw", (0, 215, 255)),
            ("Bike / Two-Wheeler", (0, 140, 255)),
            ("Car", (255, 191, 0)),
            ("Bus", (255, 0, 255)),
            ("Truck", (50, 50, 255)),
            ("Tempo / LCV", (0, 255, 127)),
        ]
        for cname, col in top_classes:
            cnt = self.line_counter.class_counts.get(cname, 0)
            cv2.putText(frame, f"{cname}:", (hud_x + 12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, str(cnt), (hud_x + hud_w - 45, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 2, cv2.LINE_AA)
            y_offset += 20

        return frame

    def process_video(self, video_source: str, interactive: bool = True, max_frames: Optional[int] = None):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[!] Error: Could not open video source: {video_source}")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._init_line_counter(fw, fh)

        writer = None
        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            writer = cv2.VideoWriter(self.output_path, fourcc, fps_in, (fw, fh))

        window_name = "Indian Traffic AI - Vehicle Detection & Virtual Line Counter"
        if interactive:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1280, fw), min(720, fh))
            cv2.setMouseCallback(window_name, self.mouse_callback)

        frame_idx = 0
        fps_curr = 0.0
        t0 = time.time()
        self.start_time = t0
        paused = False

        try:
            while cap.isOpened():
                if max_frames and frame_idx >= max_frames:
                    break

                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    t_frame = time.time()

                    results = self.model.track(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        tracker=self.tracker,
                        persist=True,
                        verbose=False,
                        device=self.device,
                        half=self.use_fp16
                    )

                    active_ids = set()
                    if results and results[0].boxes and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        ids = results[0].boxes.id.int().cpu().numpy()
                        classes = results[0].boxes.cls.int().cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()

                        for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                            active_ids.add(int(track_id))
                            raw_name = results[0].names.get(int(cls_id), "Vehicle")
                            self.line_counter.update_vehicle(
                                track_id=int(track_id),
                                bbox=tuple(box.astype(int)),
                                raw_class_name=raw_name,
                                conf=float(conf),
                                frame_idx=frame_idx
                            )

                            category = map_class_name(raw_name)
                            color = get_category_color(category)
                            x1, y1, x2, y2 = box.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                            label = f"#{track_id} {category} {conf:.2f}"
                            cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

                    if self.show_trails:
                        self.line_counter.draw_trajectories(frame, active_ids)

                    self.line_counter.draw_line(frame)

                    dt = time.time() - t_frame
                    if dt > 0:
                        fps_curr = 0.9 * fps_curr + 0.1 * (1.0 / dt)

                    self.draw_hud(frame, fps_curr)

                    if writer:
                        writer.write(frame)

                if interactive:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in [ord('q'), 27]:
                        break
                    elif key in [ord('p'), 32]:
                        paused = not paused
                    elif key == ord('r'):
                        self.line_counter.reset()
                    elif key == ord('s'):
                        snap_path = f"snapshot_frame_{frame_idx}.jpg"
                        cv2.imwrite(snap_path, frame)
                        print(f"[*] Snapshot saved: {snap_path}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if interactive:
                cv2.destroyAllWindows()

        self.total_frames_processed = frame_idx
        elapsed = time.time() - self.start_time
        summary = self.get_summary()
        summary["process_time"] = elapsed
        summary["fps"] = round(frame_idx / max(elapsed, 0.001), 1)
        self.print_summary(summary)

        if self.save_csv and self.line_counter:
            df = pd.DataFrame(self.line_counter.events)
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            df.to_csv(self.csv_path, index=False)
            print(f"[*] Audit CSV log saved to: {self.csv_path}")

    def get_summary(self) -> Dict:
        if self.line_counter is None:
            return {}
        return {
            "total_count": self.line_counter.total_count,
            "inward_count": self.line_counter.direction_counts.get("Inward", 0),
            "outward_count": self.line_counter.direction_counts.get("Outward", 0),
            "class_counts": dict(self.line_counter.class_counts),
            "device_name": self.device_name,
            "total_frames": self.total_frames_processed
        }

    def print_summary(self, summary: Dict):
        print("\n" + "=" * 60)
        print("         INDIAN TRAFFIC AI - COUNTING AUDIT SUMMARY          ")
        print("=" * 60)
        print(f"  Total Vehicles Counted : {summary.get('total_count', 0)}")
        print(f"  Inward Flow (Towards)  : {summary.get('inward_count', 0)}")
        print(f"  Outward Flow (Away)    : {summary.get('outward_count', 0)}")
        print(f"  Frames Processed       : {summary.get('total_frames', 0)}")
        print(f"  Hardware Engine        : {summary.get('device_name', 'N/A')}")
        print("-" * 60)
        print("  Breakdown by Indian Class:")
        for cls_name, cnt in summary.get("class_counts", {}).items():
            print(f"    - {cls_name:<20}: {cnt}")
        print("=" * 60 + "\n")


# =============================================================================
# FLASK WEB DASHBOARD & REST API SERVER
# =============================================================================

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def to_serializable(obj):
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


app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.json = NumpyJSONProvider(app)

# In-memory jobs & workers
jobs = {}
jobs_lock = threading.Lock()

gpu_workers = {}
gpu_workers_lock = threading.Lock()
pending_worker_jobs = []


def get_active_workers_list():
    now = time.time()
    active = []
    with gpu_workers_lock:
        for w_id, w in list(gpu_workers.items()):
            if now - w["last_heartbeat"] <= 15.0:
                active.append({
                    "id": w["id"],
                    "name": w["name"],
                    "hostname": w["hostname"],
                    "gpu_name": w["gpu_name"],
                    "cuda_version": w["cuda_version"],
                    "vram_gb": w["vram_gb"],
                    "status": w["status"],
                    "jobs_completed": w["jobs_completed"]
                })
            else:
                w["status"] = "offline"
    return active


def get_available_worker():
    now = time.time()
    with gpu_workers_lock:
        for w_id, w in gpu_workers.items():
            if now - w["last_heartbeat"] <= 15.0 and w["status"] == "idle":
                return w_id
    return None


def init_compute_device():
    has_cuda = torch.cuda.is_available()
    device = "cpu"
    device_name = "CPU (Multi-Core)"
    supports_fp16 = False

    if has_cuda:
        try:
            device = "cuda:0"
            device_name = torch.cuda.get_device_name(0)
            test_tensor = torch.zeros((1, 1), device="cuda:0") + 1.0
            del test_tensor
            try:
                with torch.no_grad():
                    test_conv = torch.nn.functional.conv2d(
                        torch.zeros((1, 1, 3, 3), device="cuda:0"),
                        torch.zeros((1, 1, 3, 3), device="cuda:0")
                    )
                    del test_conv
            except Exception:
                torch.backends.cudnn.enabled = False
            torch.cuda.empty_cache()
            supports_fp16 = True
        except Exception as e:
            has_cuda = False
            device = "cpu"
            device_name = "CPU Mode (GPU Incompatible)"
            supports_fp16 = False

    return has_cuda, device, device_name, supports_fp16


HAS_CUDA, DEVICE, DEVICE_NAME, SUPPORTS_FP16 = init_compute_device()


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"width": w, "height": h, "fps": round(fps, 1), "frames": frames, "total_frames": frames}


def extract_first_frame(video_path, output_jpg_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(output_jpg_path, frame)
        return True
    return False


def draw_frame_hud(frame, counter, fps_curr, active_device=DEVICE):
    fh, fw = frame.shape[:2]
    hud_w = 320
    hud_h = 240
    hud_x = fw - hud_w - 15
    hud_y = 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 191, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "INDIAN TRAFFIC AI", (hud_x + 12, hud_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    fps_color = (0, 255, 0) if fps_curr >= 25 else (0, 215, 255)
    cv2.putText(frame, f"{fps_curr:.1f} FPS", (hud_x + hud_w - 75, hud_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"TOTAL: {counter.total_count}",
                (hud_x + 12, hud_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)

    in_cnt = counter.direction_counts.get("Inward", 0)
    out_cnt = counter.direction_counts.get("Outward", 0)
    cv2.putText(frame, f"IN: {in_cnt}", (hud_x + 12, hud_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 127), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT: {out_cnt}", (hud_x + 110, hud_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 140, 255), 2, cv2.LINE_AA)

    y_off = hud_y + 102
    for cname in ["Auto-Rickshaw", "Bike / Two-Wheeler", "Car", "Bus", "Truck", "Tempo / LCV"]:
        cnt = counter.class_counts.get(cname, 0)
        col = get_category_color(cname)
        cv2.putText(frame, f"{cname}:", (hud_x + 12, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, str(cnt), (hud_x + hud_w - 40, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 2, cv2.LINE_AA)
        y_off += 20

    return frame


def run_video_job(job_id, video_path, line_coords, conf_thresh):
    """Background worker processing video on NVIDIA GPU or CPU."""
    try:
        t_start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = f"Failed to open video: {video_path}"
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_dim = max(w, h)
        scale = 1920.0 / max_dim if max_dim > 1920 else 1.0
        proc_w = int(w * scale)
        proc_h = int(h * scale)

        p1 = (int(line_coords[0] * scale), int(line_coords[1] * scale))
        p2 = (int(line_coords[2] * scale), int(line_coords[3] * scale))
        counter = LineCounter(pt1=p1, pt2=p2)

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
                print(f"[!] Warning: CUDA execution fallback to CPU ({e})")
                current_device = "cpu"
                model.to("cpu")
                model.model.float()
        else:
            current_device = "cpu"

        web_video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
        excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
        csv_path = os.path.join(OUTPUTS_DIR, f"log_{job_id}.csv")

        ff_exe = None
        if imageio_ffmpeg is not None:
            try:
                ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ff_exe = None
        if not ff_exe:
            ff_exe = "ffmpeg"

        # Try FFmpeg H.264 pipe for direct browser playback, fallback to OpenCV VideoWriter
        pipe_out = None
        cv_writer = None
        try:
            out_cmd = [
                ff_exe,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{proc_w}x{proc_h}",
                "-pix_fmt", "bgr24",
                "-r", str(round(src_fps, 2)),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "24",
                "-movflags", "+faststart",
                web_video_path
            ]
            pipe_out = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pipe_out = None
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            cv_writer = cv2.VideoWriter(web_video_path, fourcc, src_fps, (proc_w, proc_h))

        frame_idx = 0
        fps_curr = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_f0 = time.time()

            if scale != 1.0:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            results = model.track(
                source=frame,
                conf=conf_thresh,
                iou=0.45,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                device=current_device,
                half=(SUPPORTS_FP16 and current_device.startswith("cuda"))
            )

            active_ids = set()
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                    active_ids.add(int(track_id))
                    raw_name = results[0].names.get(int(cls_id), "Vehicle")
                    counter.update_vehicle(
                        track_id=int(track_id),
                        bbox=tuple(box.astype(int)),
                        raw_class_name=raw_name,
                        conf=float(conf),
                        frame_idx=frame_idx
                    )

                    category = map_class_name(raw_name)
                    color = get_category_color(category)
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    label = f"#{track_id} {category} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

            counter.draw_trajectories(frame, active_ids)
            counter.draw_line(frame)

            dt = time.time() - t_f0
            if dt > 0:
                fps_curr = 0.9 * fps_curr + 0.1 * (1.0 / dt)

            draw_frame_hud(frame, counter, fps_curr, active_device=current_device)

            if pipe_out and pipe_out.stdin:
                try:
                    pipe_out.stdin.write(frame.tobytes())
                except Exception:
                    pass
            elif cv_writer:
                cv_writer.write(frame)

            if frame_idx % 8 == 0 or frame_idx == total_frames:
                elapsed_sec = time.time() - t_start
                pct = int((frame_idx / max(total_frames, 1)) * 100)
                eta = round(((total_frames - frame_idx) / max(fps_curr, 0.1)), 1) if fps_curr > 0 else 0
                with jobs_lock:
                    jobs[job_id]["percent"] = pct
                    jobs[job_id]["current_frame"] = frame_idx
                    jobs[job_id]["total_frames"] = total_frames
                    jobs[job_id]["fps"] = round(fps_curr, 1)
                    jobs[job_id]["elapsed_sec"] = round(elapsed_sec, 1)
                    jobs[job_id]["eta_sec"] = eta
                    jobs[job_id]["total_count"] = counter.total_count
                    jobs[job_id]["active_counts"] = dict(counter.class_counts)

        cap.release()
        if pipe_out and pipe_out.stdin:
            try:
                pipe_out.stdin.close()
                pipe_out.wait(timeout=15)
            except Exception:
                pass
        if cv_writer:
            cv_writer.release()

        elapsed_total = time.time() - t_start
        summary_data = {
            "total_count": counter.total_count,
            "inward_count": counter.direction_counts.get("Inward", 0),
            "outward_count": counter.direction_counts.get("Outward", 0),
            "class_counts": dict(counter.class_counts),
            "inward_class_counts": {},
            "outward_class_counts": {},
            "video_name": os.path.basename(video_path),
            "fps": round(frame_idx / max(elapsed_total, 0.001), 1),
            "total_frames": frame_idx,
            "device_name": DEVICE_NAME if current_device.startswith("cuda") else "CPU Multi-Threading",
            "process_time": elapsed_total
        }

        for ev in counter.events:
            c = ev["class"]
            d = ev["direction"]
            if d == "Inward":
                summary_data["inward_class_counts"][c] = summary_data["inward_class_counts"].get(c, 0) + 1
            elif d == "Outward":
                summary_data["outward_class_counts"][c] = summary_data["outward_class_counts"].get(c, 0) + 1

        try:
            generate_traffic_excel(counter.events, summary_data, excel_path)
        except Exception as e:
            print(f"[!] Excel generation warning: {e}")

        try:
            df = pd.DataFrame(counter.events)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["percent"] = 100
            jobs[job_id]["summary"] = summary_data
            jobs[job_id]["events"] = counter.events
            jobs[job_id]["web_video_path"] = web_video_path
            jobs[job_id]["excel_path"] = excel_path
            jobs[job_id]["csv_path"] = csv_path

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        print(f"[!] Job {job_id} error: {e}")


# =============================================================================
# FLASK WEB ROUTES & ENDPOINTS
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample-video", methods=["GET", "POST"])
@app.route("/api/sample_video", methods=["GET", "POST"])
def select_sample_video():
    """Sets the bundled sample video as active source."""
    if not os.path.exists(SAMPLE_VIDEO):
        return jsonify({"status": "error", "message": "Sample video file not found."}), 404

    info = get_video_info(SAMPLE_VIDEO) or {"width": 1280, "height": 720, "fps": 25.0, "total_frames": 100}
    frame_name = "sample_preview_frame.jpg"
    frame_path = os.path.join(UPLOADS_DIR, frame_name)
    extract_first_frame(SAMPLE_VIDEO, frame_path)
    
    try:
        extract_first_frame(SAMPLE_VIDEO, os.path.join(STATIC_DIR, frame_name))
    except Exception:
        pass

    frame_url = f"/api/frame/{frame_name}"
    return jsonify({
        "status": "ok",
        "video_path": SAMPLE_VIDEO,
        "filename": os.path.basename(SAMPLE_VIDEO),
        "width": info.get("width", 1280),
        "height": info.get("height", 720),
        "fps": info.get("fps", 25.0),
        "total_frames": info.get("total_frames", info.get("frames", 100)),
        "frame_url": frame_url,
        "preview_url": frame_url,
        "info": info
    })


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Handles user video upload."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    saved_filename = f"user_upload_{unique_id}{ext}"
    save_path = os.path.join(UPLOADS_DIR, saved_filename)
    file.save(save_path)

    info = get_video_info(save_path) or {"width": 1280, "height": 720, "fps": 25.0, "total_frames": 100}

    frame_name = f"frame_{unique_id}.jpg"
    frame_path = os.path.join(UPLOADS_DIR, frame_name)
    extract_first_frame(save_path, frame_path)
    
    try:
        extract_first_frame(save_path, os.path.join(STATIC_DIR, frame_name))
    except Exception:
        pass

    frame_url = f"/api/frame/{frame_name}"
    return jsonify({
        "status": "ok",
        "video_path": save_path,
        "filename": file.filename,
        "width": info.get("width", 1280),
        "height": info.get("height", 720),
        "fps": info.get("fps", 25.0),
        "total_frames": info.get("total_frames", info.get("frames", 100)),
        "frame_url": frame_url,
        "preview_url": frame_url,
        "info": info
    })


@app.route("/api/frame/<path:filename>")
def serve_frame(filename):
    """Serves preview road frames."""
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "Frame not found", 404


@app.route("/api/process", methods=["POST"])
@app.route("/api/start", methods=["POST"])
def start_process():
    data = request.get_json() or {}
    video_path = data.get("video_source") or data.get("video_path")

    if not video_path or not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "Invalid video path"}), 400

    raw_line = data.get("line") or data.get("line_coords")
    line = [100, 450, 1180, 450]
    if isinstance(raw_line, str):
        try:
            parts = [int(p.strip()) for p in raw_line.split(",")]
            if len(parts) == 4:
                line = parts
        except Exception:
            pass
    elif isinstance(raw_line, (list, tuple)) and len(raw_line) == 4:
        line = [int(p) for p in raw_line]

    conf = float(data.get("conf") or data.get("conf_threshold") or 0.35)
    job_id = str(uuid.uuid4())[:8]

    info = get_video_info(video_path)
    total_frames = info.get("total_frames", info.get("frames", 100)) if info else 100

    # Check for connected remote GPU worker nodes
    worker_id = get_available_worker()
    if worker_id:
        with jobs_lock:
            jobs[job_id] = {
                "id": job_id,
                "video_source": video_path,
                "line_coords": line,
                "conf_thresh": conf,
                "status": "pending_worker",
                "percent": 0,
                "total_count": 0,
                "fps": 0.0,
                "assigned_worker": worker_id
            }
        with gpu_workers_lock:
            gpu_workers[worker_id]["status"] = "busy"
            gpu_workers[worker_id]["current_job_id"] = job_id
        pending_worker_jobs.append(job_id)
        return jsonify({"status": "ok", "job_id": job_id, "mode": "client_gpu_fleet", "worker_id": worker_id})

    # Local processing
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "video_source": video_path,
            "line_coords": line,
            "conf_thresh": conf,
            "status": "processing",
            "percent": 0,
            "current_frame": 0,
            "total_frames": total_frames,
            "fps": 0.0,
            "total_count": 0,
            "active_counts": {},
            "elapsed_sec": 0,
            "eta_sec": 0
        }

    threading.Thread(target=run_video_job, args=(job_id, video_path, line, conf), daemon=True).start()
    return jsonify({"status": "ok", "job_id": job_id, "mode": "local"})


@app.route("/api/progress/<job_id>")
def get_progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        return jsonify(to_serializable(job))


@app.route("/api/results/<job_id>")
def get_results(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        if job.get("status") != "completed":
            return jsonify({"status": job.get("status", "unknown")})
        return jsonify({
            "status": "completed",
            "summary": to_serializable(job.get("summary", {})),
            "events": to_serializable(job.get("events", [])),
            "video_url": f"/api/video/{job_id}",
            "excel_url": f"/api/download/excel/{job_id}",
            "csv_url": f"/api/download/csv/{job_id}"
        })


@app.route("/api/video/<job_id>")
def serve_video(job_id):
    web_video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
    if os.path.exists(web_video_path):
        return send_file(web_video_path, mimetype="video/mp4")
    return "Video not found", 404


@app.route("/api/download/excel/<job_id>")
def download_excel(job_id):
    excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
    if os.path.exists(excel_path):
        return send_file(
            excel_path,
            as_attachment=True,
            download_name=f"Traffic_Report_{job_id}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    return "Report not found", 404


@app.route("/api/download/csv/<job_id>")
def download_csv(job_id):
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


# GPU Worker Fleet endpoints
@app.route("/api/worker/register", methods=["POST"])
def worker_register():
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
    return jsonify({"status": "ok", "worker_id": w_id})


@app.route("/api/worker/heartbeat", methods=["POST"])
def worker_heartbeat():
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
        return jsonify({"status": "need_registration"}), 404


@app.route("/api/worker/job/poll", methods=["GET"])
def worker_poll_job():
    w_id = request.args.get("worker_id")
    if not w_id:
        return jsonify({"status": "error", "message": "worker_id required"}), 400
    with gpu_workers_lock:
        if w_id not in gpu_workers:
            return jsonify({"status": "need_registration"}), 404
        gpu_workers[w_id]["last_heartbeat"] = time.time()
        job_to_assign = pending_worker_jobs.pop(0) if pending_worker_jobs else None

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
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or not os.path.exists(job.get("video_source", "")):
            return "Video not found", 404
        return send_file(job.get("video_source"), mimetype="video/mp4")


@app.route("/api/worker/job/<job_id>/progress", methods=["POST"])
def worker_report_progress(job_id):
    data = request.get_json() or {}
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error"}), 404
        job.update({
            "percent": data.get("percent", job.get("percent", 0)),
            "current_frame": data.get("current_frame", job.get("current_frame", 0)),
            "total_frames": data.get("total_frames", job.get("total_frames", 100)),
            "fps": data.get("fps", job.get("fps", 0.0)),
            "elapsed_sec": data.get("elapsed_sec", 0.0),
            "eta_sec": data.get("eta_sec", 0.0),
            "total_count": data.get("total_count", job.get("total_count", 0)),
            "active_counts": data.get("active_counts", {}),
            "status": "processing"
        })
    return jsonify({"status": "ok"})


@app.route("/api/worker/job/<job_id>/complete", methods=["POST"])
def worker_job_complete(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "error"}), 404

    web_video_path = os.path.join(OUTPUTS_DIR, f"web_{job_id}.mp4")
    if "video" in request.files:
        request.files["video"].save(web_video_path)

    events = json.loads(request.form.get("events", "[]")) if "events" in request.form else []
    summary_data = json.loads(request.form.get("summary", "{}")) if "summary" in request.form else {}

    excel_path = os.path.join(OUTPUTS_DIR, f"report_{job_id}.xlsx")
    csv_path = os.path.join(OUTPUTS_DIR, f"log_{job_id}.csv")

    try:
        generate_traffic_excel(events, summary_data, excel_path)
    except Exception:
        pass

    try:
        pd.DataFrame(events).to_csv(csv_path, index=False)
    except Exception:
        pass

    with jobs_lock:
        job.update({
            "status": "completed",
            "percent": 100,
            "summary": summary_data,
            "events": events,
            "web_video_path": web_video_path,
            "excel_path": excel_path,
            "csv_path": csv_path
        })
        assigned_worker = job.get("assigned_worker")

    with gpu_workers_lock:
        if assigned_worker and assigned_worker in gpu_workers:
            gpu_workers[assigned_worker]["status"] = "idle"
            gpu_workers[assigned_worker]["current_job_id"] = None
            gpu_workers[assigned_worker]["jobs_completed"] += 1

    return jsonify({"status": "ok", "job_id": job_id})


@app.route("/api/workers", methods=["GET"])
def get_workers():
    return jsonify({"status": "ok", "workers": get_active_workers_list()})


@app.route("/api/update/status")
def update_status():
    return jsonify({"status": "ok", "current_version": "1.0.0", "auto_check_on_launch": False})


@app.route("/api/update/check", methods=["GET", "POST"])
def check_update_route():
    return jsonify({"status": "ok", "update_available": False, "current_version": "1.0.0"})


@app.route("/api/update/progress")
def update_progress_route():
    return jsonify({"percent": 0, "status": "idle", "error": None})


# =============================================================================
# CLI LAUNCHER & EXECUTION RUNNERS
# =============================================================================

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
            test_tensor = torch.zeros((1, 1), device="cuda:0") + 1.0
            del test_tensor
            try:
                with torch.no_grad():
                    test_conv = torch.nn.functional.conv2d(
                        torch.zeros((1, 1, 3, 3), device="cuda:0"),
                        torch.zeros((1, 1, 3, 3), device="cuda:0")
                    )
                    del test_conv
            except Exception:
                torch.backends.cudnn.enabled = False
            torch.cuda.empty_cache()

            free_b, tot_b = torch.cuda.mem_get_info(0)
            free_gb = round(free_b / (1024 ** 3), 2)
            tot_gb = round(tot_b / (1024 ** 3), 2)
            vram_str = f"{tot_gb:.2f} GB ({free_gb:.2f} GB Free)"
            compute_mode = "Hardware CUDA Acceleration (FP16 Tensor Cores)"
        except Exception:
            has_cuda = False
            device_name = "CPU Mode (GPU Incompatible)"
            compute_mode = "CPU Multi-Threading (GPU Fallback)"

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
    print("-" * 72)


def launch_web_server(port=5000, open_browser=True):
    """Launches the Flask Web Studio dashboard."""
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
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


def launch_interactive_window(video_path=None, weights=None, line=None, conf=0.35):
    """Launches the OpenCV desktop window with mouse drag-and-drop line."""
    if not video_path:
        sample_path = SAMPLE_VIDEO
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

    weights_file = weights or WEIGHTS_PATH
    system = VehicleCounterSystem(
        weights_path=weights_file,
        conf_threshold=conf,
        line_coords=line_coords,
        save_output=True,
        output_path=os.path.join(OUTPUTS_DIR, "interactive_output.mp4"),
        save_csv=True,
        csv_path=os.path.join(OUTPUTS_DIR, "interactive_log.csv")
    )
    print(f"\n[*] Starting interactive OpenCV video window for: {video_path}")
    print("[*] Controls:")
    print("    - Drag yellow circular handles with mouse to adjust virtual counting line")
    print("    - Press 'p' or SPACE to Pause / Resume")
    print("    - Press 'r' to Reset vehicle counters")
    print("    - Press 's' to Save a frame snapshot")
    print("    - Press 'q' or ESC to Exit\n")
    system.process_video(video_source=video_path, interactive=True)


def launch_batch_processing(video_path, output_path=None, line=None, conf=0.35, excel=True, max_frames=None):
    """Headless high-speed video batch processor."""
    if not output_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUTS_DIR, f"{base}_counted.mp4")

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

    system = VehicleCounterSystem(
        weights_path=WEIGHTS_PATH,
        conf_threshold=conf,
        line_coords=line_coords,
        save_output=True,
        output_path=output_path,
        save_csv=True,
        csv_path=csv_path
    )
    print(f"\n[*] Running Batch Video Processing:")
    print(f"    - Source Video : {video_path}")
    print(f"    - Output Video : {output_path}")
    print(f"    - Output CSV   : {csv_path}")

    t0 = time.time()
    system.process_video(video_source=video_path, interactive=False, max_frames=max_frames)
    elapsed = time.time() - t0

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
                "total_frames": system.total_frames_processed,
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
            print(f"    - Output Excel : {excel_path}")
        except Exception as e:
            print(f"[!] Warning: Failed to generate Excel report: {e}")

    print(f"\n[OK] Video processing completed in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Indian Traffic AI - Vehicle Detection & Counting System")
    parser.add_argument("--web", action="store_true", help="Launch Web Browser Dashboard Studio (Default)")
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
    parser.add_argument("--select", action="store_true", help="Open Windows file picker dialog to choose a video")

    args = parser.parse_args()

    hw = get_hardware_info()
    print_banner(hw)

    def find_sample_video():
        for p in [
            os.path.join(APP_DIR, "compressed_output_TrafficPolice.mp4"),
            os.path.join(RESOURCE_DIR, "compressed_output_TrafficPolice.mp4"),
            "compressed_output_TrafficPolice.mp4"
        ]:
            if os.path.exists(p):
                return p
        return "compressed_output_TrafficPolice.mp4"

    # Interactive OpenCV desktop window mode
    if args.interactive or args.select:
        video_src = args.source
        if not video_src and not args.select:
            video_src = find_sample_video()
        launch_interactive_window(
            video_path=video_src,
            line=args.line,
            conf=args.conf
        )
        return

    # Headless batch processing mode
    if args.batch:
        video_src = args.source
        if not video_src:
            video_src = find_sample_video()
        if not os.path.exists(video_src):
            print(f"[!] Error: Video file not found: {video_src}")
            return
        launch_batch_processing(
            video_path=video_src,
            output_path=args.output,
            line=args.line,
            conf=args.conf,
            excel=True,
            max_frames=args.max_frames
        )
        return

    # Default Mode: Full Web Studio Dashboard
    launch_web_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

