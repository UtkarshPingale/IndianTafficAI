"""
Vehicle Classes & Color Mappings for Indian Traffic Detection & Counting.
Supports:
1. IISc Bangalore UVH-26 Indian Traffic Model (14 classes including Auto-Rickshaw, 2-Wheeler, etc.)
2. Standard YOLO / COCO vehicle classes (fallback)
"""

from typing import Dict, Tuple, Optional

# UVH-26 (Indian Traffic) fine-grained class list
UVH_CLASSES = [
    "Hatchback",       # 0
    "Sedan",           # 1
    "SUV",             # 2
    "MUV",             # 3
    "Bus",             # 4
    "Truck",           # 5
    "Three-wheeler",   # 6 (Auto-rickshaw)
    "Two-wheeler",     # 7 (Bike / Motorcycle / Scooter)
    "LCV",             # 8 (Light Commercial Vehicle / Tempo)
    "Mini-bus",        # 9
    "Tempo-traveller",  # 10
    "Bicycle",         # 11
    "Van",             # 12
    "Others"           # 13
]

# High-level aggregated categories for Indian Traffic
# Simplifies 14 fine classes into key categories requested by user:
# Auto-Rickshaw, Bike/Two-Wheeler, Car, Truck, Bus, Tempo/LCV, Bicycle, Other
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

# Standard COCO fallback mapping
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    1: "Bicycle",
    2: "Car",
    3: "Bike / Two-Wheeler",
    5: "Bus",
    7: "Truck"
}

# Distinct BGR color palette for visualization
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

# Category emoji / icons for display
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
    # Check direct mapping
    if raw_name in UVH_TO_CATEGORY:
        return UVH_TO_CATEGORY[raw_name]
    
    # Case-insensitive checks
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
