"""
Virtual Line Counter Engine with Direction Detection, Deduplication, and Trajectory Tracking.
"""

import math
import time
from collections import defaultdict, deque, Counter
from typing import Dict, List, Optional, Tuple, Set
import cv2
import numpy as np

from vehicle_classes import map_class_name, get_category_color


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Return True if points A, B, C are in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float]
) -> bool:
    """
    Check if line segment AB intersects line segment CD.
    AB is the virtual counting line, CD is the vehicle trajectory vector.
    """
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class LineCounter:
    """
    Tracks vehicle centroids across a virtual line and counts them per class and direction.
    """

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

        # Track history: track_id -> deque of (cx, cy)
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_length))

        # Vehicle class cache for each track_id (keeps highest confidence or most frequent class)
        self.track_classes: Dict[int, str] = {}
        self.track_confs: Dict[int, float] = {}

        # Set of track IDs that have already been counted to prevent duplicates
        self.counted_ids: Set[int] = set()

        # Cumulative statistics
        self.class_counts: Counter = Counter()
        self.direction_counts: Counter = Counter({"Inward": 0, "Outward": 0})
        self.total_count: int = 0

        # Detailed event log: list of dicts
        self.events: List[Dict] = []

        # Visual feedback: flash counter when a crossing occurs
        self.flash_frames = 0
        self.last_crossed_vehicle = ""

    def set_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """Update line coordinates."""
        self.pt1 = list(pt1)
        self.pt2 = list(pt2)

    def side_of_line(self, point: Tuple[float, float]) -> float:
        """
        Computes the signed 2D cross product of vector (pt2 - pt1) and (point - pt1).
        Sign indicates which side of the line the point is on.
        """
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
        """
        Update a vehicle's position, trajectory, and check for line crossing.

        Args:
            track_id: Persistent ID from YOLO tracker.
            bbox: (x1, y1, x2, y2)
            raw_class_name: Raw YOLO class name.
            conf: Detection confidence.
            frame_idx: Current frame index.

        Returns:
            Event dict if a new crossing was counted, else None.
        """
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

            # Check if trajectory segment prev_pt -> curr_pt intersects the virtual line
            L1 = (float(self.pt1[0]), float(self.pt1[1]))
            L2 = (float(self.pt2[0]), float(self.pt2[1]))
            P1 = (float(prev_pt[0]), float(prev_pt[1]))
            P2 = (float(curr_pt[0]), float(curr_pt[1]))

            if segments_intersect(L1, L2, P1, P2):
                # Calculate crossing direction
                side_prev = self.side_of_line(prev_pt)
                side_curr = self.side_of_line(curr_pt)

                # Direction convention:
                # side_prev < 0 and side_curr >= 0 => Inward / Down / Right
                # side_prev > 0 and side_curr <= 0 => Outward / Up / Left
                if side_prev < 0:
                    direction = "Inward"
                else:
                    direction = "Outward"

                # Mark as counted
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
        """Draw the virtual counting line and glowing endpoints onto the frame."""
        p1 = (int(self.pt1[0]), int(self.pt1[1]))
        p2 = (int(self.pt2[0]), int(self.pt2[1]))

        # Flash effect when a vehicle crosses
        if self.flash_frames > 0:
            color = self.flash_color
            thick = self.line_thickness + 3
            self.flash_frames -= 1
        else:
            color = self.line_color
            thick = self.line_thickness

        # Outer glow
        cv2.line(frame, p1, p2, (0, 100, 100), thick + 4, cv2.LINE_AA)
        # Main line
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)

        # Draw handles at endpoints for mouse dragging
        for pt in [p1, p2]:
            cv2.circle(frame, pt, 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)

        # Label above the center of the line
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        label = "VIRTUAL COUNTING LINE"
        cv2.putText(
            frame,
            label,
            (mid_x - 90, max(25, mid_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_trajectories(self, frame: np.ndarray, active_ids: Optional[Set[int]] = None) -> np.ndarray:
        """Draw motion breadcrumb trails for actively tracked vehicles."""
        targets = [tid for tid in self.track_history if (active_ids is None or tid in active_ids)]
        for track_id in targets:
            pts = self.track_history[track_id]
            if len(pts) < 2:
                continue
            cat = self.track_classes.get(track_id, "Vehicle")
            color = get_category_color(cat)

            # Draw smooth fading trail
            num_pts = len(pts)
            for i in range(1, num_pts):
                thickness = int(math.sqrt(64 / float(num_pts - i + 1)))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Prune stale tracks to maintain fast processing
        if len(self.track_history) > 120 and active_ids is not None:
            stale_keys = [k for k in self.track_history.keys() if k not in active_ids and k in self.counted_ids]
            for k in stale_keys:
                del self.track_history[k]

        return frame

    def reset(self):
        """Reset all counters and state."""
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
