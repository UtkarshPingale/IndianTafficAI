"""
Streamlit Web Dashboard for Indian Vehicle Detection, Tracking, and Virtual Line Counting.
Hardware accelerated for NVIDIA GeForce RTX 4070 SUPER.
"""

import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
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

# Page configuration
st.set_page_config(
    page_title="Indian Vehicle Counter AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3b 100%);
        border: 1px solid #363d4e;
        border-radius: 10px;
        padding: 15px 20px;
        color: white;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
        margin-top: -5px;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .gpu-badge {
        display: inline-block;
        padding: 4px 10px;
        background: #00e676;
        color: #000;
        font-weight: bold;
        border-radius: 5px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_yolo_model(weights_path: str):
    """Cache YOLO model loading with CUDA optimization."""
    has_cuda = torch.cuda.is_available()
    device = "cuda:0" if has_cuda else "cpu"
    if has_cuda:
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(weights_path):
        weights_path = "yolo11s.pt"
    model = YOLO(weights_path)
    if has_cuda:
        try:
            model.predict(np.zeros((384, 640, 3), dtype=np.uint8), device=device, half=True, verbose=False)
        except Exception:
            pass
    return model, device


def main():
    st.title("🚗 Indian Vehicle Detection & Virtual Line Counter")
    
    # GPU status banner
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    st.markdown(
        f"<span class='gpu-badge'>⚡ {'CUDA ACTIVE: ' + device_name if has_cuda else 'RUNNING ON CPU'}</span> "
        f"<span style='color: #888; font-size: 0.85rem; margin-left: 10px;'>YOLOv11 + ByteTrack Tracking Engine</span>",
        unsafe_allow_html=True
    )
    st.write("")

    # Sidebar settings
    st.sidebar.header("⚙️ Configuration")

    # Model choice
    model_options = {
        "Indian Traffic Model (UVH-26: Auto, Bike, Car, Truck, Bus, Tempo)": "models/yolo11s_indian_traffic.pt",
        "Standard YOLOv11s (COCO Baseline)": "yolo11s.pt",
        "Custom Upload / Path": "custom"
    }
    selected_model_label = st.sidebar.selectbox("Model Architecture", list(model_options.keys()))
    if selected_model_label == "Custom Upload / Path":
        weights_path = st.sidebar.text_input("Enter weights path (.pt):", "models/yolo11s_indian_traffic.pt")
    else:
        weights_path = model_options[selected_model_label]

    # Video Source
    st.sidebar.subheader("📹 Video Source")
    demo_video = "compressed_output_TrafficPolice.mp4"
    source_type = st.sidebar.radio("Choose Input:", ["Demo Video (Traffic Police)", "Upload Custom Video"])

    temp_video_path = None
    if source_type == "Demo Video (Traffic Police)":
        if os.path.exists(demo_video):
            temp_video_path = demo_video
        else:
            st.sidebar.warning(f"Demo video '{demo_video}' not found in folder.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

    # Detection parameters
    st.sidebar.subheader("🎯 Detection Parameters")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
    iou_thresh = st.sidebar.slider("IOU Threshold", 0.20, 0.80, 0.45, 0.05)
    show_trails = st.sidebar.checkbox("Show Motion Trails", value=True)

    # Virtual Line Settings
    st.sidebar.subheader("📍 Virtual Counting Line")
    line_preset = st.sidebar.selectbox(
        "Line Position Preset",
        ["Horizontal (Lower 60%)", "Horizontal (Middle 50%)", "Diagonal Crossing", "Custom"]
    )

    # Probe video dimensions if available
    video_w, video_h = 1280, 720
    if temp_video_path and os.path.exists(temp_video_path):
        probe_cap = cv2.VideoCapture(temp_video_path)
        if probe_cap.isOpened():
            video_w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe_cap.release()

    if line_preset == "Horizontal (Lower 60%)":
        p1 = (int(video_w * 0.05), int(video_h * 0.60))
        p2 = (int(video_w * 0.95), int(video_h * 0.60))
    elif line_preset == "Horizontal (Middle 50%)":
        p1 = (int(video_w * 0.05), int(video_h * 0.50))
        p2 = (int(video_w * 0.95), int(video_h * 0.50))
    elif line_preset == "Diagonal Crossing":
        p1 = (int(video_w * 0.10), int(video_h * 0.40))
        p2 = (int(video_w * 0.90), int(video_h * 0.75))
    else:
        col_x1, col_y1 = st.sidebar.columns(2)
        with col_x1:
            x1 = st.number_input("X1", 0, video_w, int(video_w * 0.05))
        with col_y1:
            y1 = st.number_input("Y1", 0, video_h, int(video_h * 0.60))
        col_x2, col_y2 = st.sidebar.columns(2)
        with col_x2:
            x2 = st.number_input("X2", 0, video_w, int(video_w * 0.95))
        with col_y2:
            y2 = st.number_input("Y2", 0, video_h, int(video_h * 0.60))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

    st.sidebar.info(f"Line Coords: {p1} ➡️ {p2}")

    # Main dashboard metrics placeholders
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-card'><div class='metric-lbl'>Total Counted</div><div class='metric-val' id='tot'>0</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'><div class='metric-lbl'>Auto-Rickshaws</div><div class='metric-val' id='auto' style='color:#ffd700'>0</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'><div class='metric-lbl'>Bikes / Two-Wheelers</div><div class='metric-val' id='bike' style='color:#ff8c00'>0</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-card'><div class='metric-lbl'>Cars / SUVs</div><div class='metric-val' id='car' style='color:#00bfff'>0</div></div>", unsafe_allow_html=True)

    # Process controls
    start_btn = st.button("▶️ Start Video Analysis", type="primary", use_container_width=True)

    if start_btn:
        if not temp_video_path or not os.path.exists(temp_video_path):
            st.error("Please provide or upload a valid video file first.")
            return

        # Load Model
        with st.spinner("Loading YOLOv11 model onto GPU..."):
            model, device = load_yolo_model(weights_path)
            model_classes = model.names
            is_uvh = "Three-wheeler" in list(model_classes.values()) or "Hatchback" in list(model_classes.values())
            target_ids = None if is_uvh else [c for c in COCO_VEHICLE_CLASSES.keys() if c in model_classes]

        # Initialize Line Counter
        line_counter = LineCounter(p1, p2, trail_length=25)

        # Video stream containers
        st_frame = st.empty()
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            st.subheader("📊 Vehicle Category Breakdown")
            chart_placeholder = st.empty()
        with table_col:
            st.subheader("📋 Real-time Crossing Events")
            table_placeholder = st.empty()

        cap = cv2.VideoCapture(temp_video_path)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # Processing loop
        stop_btn = st.sidebar.button("⏹️ Stop")
        progress_bar = st.progress(0.0)

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # YOLO inference with tracker
            track_args = {
                "source": frame,
                "persist": True,
                "tracker": "bytetrack.yaml",
                "conf": conf_thresh,
                "iou": iou_thresh,
                "device": device,
                "verbose": False
            }
            if target_ids is not None:
                track_args["classes"] = target_ids

            results = model.track(**track_args)[0]

            # Parse detections
            active_ids = set()
            boxes = results.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for bbox, track_id, cls_id, conf in zip(xyxy, track_ids, cls_ids, confs):
                    active_ids.add(int(track_id))
                    raw_name = model_classes.get(cls_id, "Vehicle")
                    category = map_class_name(raw_name)
                    color = get_category_color(category)

                    # Update line counter
                    line_counter.update_vehicle(
                        track_id=track_id,
                        bbox=bbox,
                        raw_class_name=raw_name,
                        conf=conf,
                        frame_idx=frame_idx
                    )

                    # Draw Bounding Box & Pill Label
                    x1_b, y1_b, x2_b, y2_b = bbox
                    cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), color, 2, cv2.LINE_AA)
                    lbl = f"#{track_id} {category} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1_b, max(0, y1_b - th - 6)), (x1_b + tw + 6, max(0, y1_b)), color, -1)
                    cv2.putText(frame, lbl, (x1_b + 3, max(0, y1_b - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if show_trails:
                line_counter.draw_trajectories(frame, active_ids=active_ids)

            line_counter.draw_line(frame)

            # Update metrics on top
            with m1:
                st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Total Counted</div><div class='metric-val'>{line_counter.total_count}</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Auto-Rickshaws</div><div class='metric-val' style='color:#ffd700'>{line_counter.class_counts.get('Auto-Rickshaw', 0)}</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Bikes / Two-Wheelers</div><div class='metric-val' style='color:#ff8c00'>{line_counter.class_counts.get('Bike / Two-Wheeler', 0)}</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Cars / SUVs</div><div class='metric-val' style='color:#00bfff'>{line_counter.class_counts.get('Car', 0)}</div></div>", unsafe_allow_html=True)

            # Render frame to streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(rgb_frame, channels="RGB", use_container_width=True)

            # Periodically update chart & table every 10 frames
            if frame_idx % 10 == 0:
                if total_frames > 0:
                    progress_bar.progress(min(1.0, frame_idx / total_frames))

                if line_counter.class_counts:
                    df_counts = pd.DataFrame(
                        list(line_counter.class_counts.items()),
                        columns=["Vehicle Category", "Count"]
                    ).sort_values(by="Count", ascending=False)
                    chart_placeholder.bar_chart(df_counts.set_index("Vehicle Category"))

                if line_counter.events:
                    df_events = pd.DataFrame(line_counter.events)[["timestamp", "track_id", "class", "direction", "confidence"]].tail(10)
                    table_placeholder.dataframe(df_events, use_container_width=True)

        cap.release()
        st.success(f"Analysis Finished! Total Vehicles Counted: {line_counter.total_count}")

        # Download CSV
        if line_counter.events:
            df_full = pd.DataFrame(line_counter.events)
            csv_data = df_full.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Detailed CSV Report",
                data=csv_data,
                file_name="traffic_counting_report.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
