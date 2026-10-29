"""
Train / Fine-tune YOLOv11 for Indian Vehicle Detection on NVIDIA RTX 4070 SUPER.

Features:
- Configured specifically for NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)
- Automatic mixed precision (AMP fp16) for Tensor Core acceleration
- Supports training from scratch or fine-tuning from YOLOv11s / YOLOv11m
- Automated dataset validation and download helper for Indian vehicle datasets
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO


def check_gpu():
    """Verify CUDA and print RTX 4070 Super specs."""
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available! Training will run on CPU, which is very slow.")
        return "cpu"

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print("=" * 60)
    print(f"  GPU Detected: {device_name}")
    print(f"  VRAM Available: {total_mem:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print("=" * 60)
    return 0


def prepare_sample_dataset_yaml(dataset_dir: str) -> str:
    """Ensure dataset.yaml exists and paths are formatted for YOLO."""
    p = Path(dataset_dir)
    yaml_path = p / "data.yaml"
    if not yaml_path.exists():
        yaml_path = p / "dataset.yaml"

    if yaml_path.exists():
        print(f"[Dataset] Found YAML configuration at '{yaml_path}'")
        return str(yaml_path)

    # If no yaml exists, construct one
    print(f"[Dataset] Creating custom dataset.yaml in '{dataset_dir}'...")
    yaml_data = {
        "path": str(p.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {
            0: "Auto-Rickshaw",
            1: "Bike / Two-Wheeler",
            2: "Car",
            3: "Truck",
            4: "Bus",
            5: "Tempo / LCV"
        }
    }
    with open(p / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return str(p / "dataset.yaml")


def train(
    dataset_yaml: str,
    base_model: str = "yolo11s.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    patience: int = 30,
    project: str = "runs/train",
    name: str = "indian_vehicle_yolo11",
    workers: int = 4
):
    """
    Train or fine-tune YOLOv11 on custom Indian vehicle dataset.
    Optimized for RTX 4070 SUPER.
    """
    device = check_gpu()

    print(f"\n[Training Config]")
    print(f"  Base Model  : {base_model}")
    print(f"  Dataset YAML: {dataset_yaml}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch Size  : {batch_size}")
    print(f"  Image Size  : {imgsz}x{imgsz}")
    print(f"  Device      : {device}")
    print(f"  Workers     : {workers}\n")

    # Load base model
    model = YOLO(base_model)

    # Launch training
    # For RTX 4070 SUPER (12GB VRAM):
    # - batch=16 with imgsz=640 fits comfortably using ~4-6GB VRAM
    # - amp=True enables fp16 mixed precision for max throughput
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        amp=True,              # Automatic Mixed Precision
        patience=patience,     # Early stopping patience
        optimizer="auto",      # Auto selects AdamW or SGD with cosine annealing
        lr0=lr0,
        lrf=0.01,
        project=project,
        name=name,
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Best Weights Saved at: {os.path.join(project, name, 'weights', 'best.pt')}")
    print("=" * 60 + "\n")

    # Run validation
    print("[Evaluation] Running validation on validation split...")
    metrics = model.val()
    print(f"Validation mAP50    : {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95 : {metrics.box.map:.4f}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Indian Vehicle Detection on RTX 4070 Super")
    parser.add_argument("--data", type=str, required=False, default=None,
                        help="Path to dataset.yaml or dataset directory")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                        help="Base model: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11x.pt")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 is ideal for 12GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--name", type=str, default="indian_traffic_yolo11", help="Experiment name")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data
    if data_path is None:
        print("[Info] No dataset specified. Looking for local datasets...")
        if os.path.exists("dataset.yaml"):
            data_path = "dataset.yaml"
        elif os.path.exists("data.yaml"):
            data_path = "data.yaml"
        else:
            print("[Error] Please specify --data <path_to_dataset_or_yaml>.")
            print("Example: python train_indian_vehicles.py --data ./my_dataset/data.yaml --model yolo11s.pt")
            sys.exit(1)

    train(
        dataset_yaml=data_path,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name
    )


if __name__ == "__main__":
    main()
