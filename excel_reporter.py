"""
Excel Report Generator for Indian Vehicle Detection & Virtual Line Counter
Generates styled multi-sheet .xlsx workbooks and clean CSV logs.
"""

import os
from datetime import datetime
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def generate_traffic_excel(events_data, summary_data, output_excel_path):
    """
    Generates a beautifully styled multi-sheet Excel file.
    
    Args:
        events_data: list of dicts with keys:
            ['frame', 'timestamp', 'track_id', 'class', 'raw_class', 'confidence', 'direction', 'point']
        summary_data: dict with keys:
            ['total_count', 'inward_count', 'outward_count', 'class_counts', 'inward_class_counts',
             'outward_class_counts', 'video_name', 'fps', 'total_frames', 'device_name', 'process_time']
        output_excel_path: path to save .xlsx
    """
    wb = openpyxl.Workbook()
    
    # ------------------ STYLES ------------------
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
    align_right = Alignment(horizontal="right", vertical="center")

    # =========================================================================
    # SHEET 1: TRAFFIC SUMMARY & METRICS
    # =========================================================================
    ws_sum = wb.active
    ws_sum.title = "Traffic Summary"
    ws_sum.views.sheetView[0].showGridLines = True
    
    # Title Banner
    ws_sum.merge_cells("A1:F1")
    title_cell = ws_sum["A1"]
    title_cell.value = "INDIAN VEHICLE DETECTION & TRAFFIC AUDIT REPORT"
    title_cell.font = font_title
    title_cell.fill = fill_dark_navy
    title_cell.alignment = align_center
    ws_sum.row_dimensions[1].height = 36
    
    # Subtitle
    ws_sum.merge_cells("A2:F2")
    sub_cell = ws_sum["A2"]
    sub_cell.value = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engine: YOLOv11 Indian Traffic Model | GPU: {summary_data.get('device_name', 'NVIDIA GPU')}"
    sub_cell.font = font_subtitle
    sub_cell.fill = fill_dark_navy
    sub_cell.alignment = align_center
    ws_sum.row_dimensions[2].height = 20

    # Section 1: Video Information
    ws_sum.cell(row=4, column=1, value="VIDEO SPECIFICATIONS").font = font_section
    info_rows = [
        ("Source Video", summary_data.get("video_name", "N/A")),
        ("Total Frames Processed", summary_data.get("total_frames", 0)),
        ("Processing Speed", f"{summary_data.get('fps', 0):.1f} FPS"),
        ("Compute Hardware", summary_data.get("device_name", "NVIDIA GeForce RTX 4070 SUPER")),
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
        
    # Section 2: Key Performance Indicators (KPIs)
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

    # Section 3: Detailed Breakdown Table by Indian Vehicle Class
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

    # Standard list of Indian classes
    all_classes = [
        "Auto-Rickshaw",
        "Bike / Two-Wheeler",
        "Car",
        "Bus",
        "Truck",
        "Tempo / LCV",
        "Bicycle",
        "Other Vehicle",
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

    # Total Summary Row
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

    # Auto-fit columns
    for col in ws_sum.iter_cols(min_row=4, max_col=6):
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len and not cell.coordinate in ("A1", "A2"):
                max_len = len(val_str)
        ws_sum.column_dimensions[col_letter].width = max(max_len + 4, 18)

    # =========================================================================
    # SHEET 2: DETAILED CROSSING EVENTS LOG
    # =========================================================================
    ws_events = wb.create_sheet(title="Crossing Events Log")
    ws_events.views.sheetView[0].showGridLines = True

    # Title
    ws_events.merge_cells("A1:I1")
    ev_title = ws_events["A1"]
    ev_title.value = "VEHICLE CROSSING AUDIT LOG (PER-VEHICLE RECORDS)"
    ev_title.font = font_title
    ev_title.fill = fill_dark_navy
    ev_title.alignment = align_center
    ws_events.row_dimensions[1].height = 32

    event_headers = [
        "S.No",
        "Frame Number",
        "Timestamp (s)",
        "Track ID",
        "Vehicle Category",
        "Raw Sub-Class",
        "Confidence (%)",
        "Flow Direction",
        "Crossing Point (X, Y)",
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

        # Direction color accent
        if ev.get("direction") == "Inward":
            c_dir.fill = fill_highlight_green
            c_dir.font = Font(name="Calibri", size=11, bold=True, color="047857")
        elif ev.get("direction") == "Outward":
            c_dir.fill = fill_highlight_yellow
            c_dir.font = Font(name="Calibri", size=11, bold=True, color="B45309")

        ws_events.row_dimensions[row_idx].height = 20
        row_idx += 1

    # Auto-fit columns for events
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
