#!/usr/bin/env python3
"""
val_yolov8.py
---------------
Validate / evaluate a trained YOLOv8 object detection model.

Example:
    python src/yolov8/val_yolov8.py \
        --weights models/yolov8/run1/weights/best.pt \
        --data_dir data/object_detection_dataset \
        --imgsz 640 \
        --project outputs/yolo_eval \
        --name tl_run1_eval

Outputs:
 - mAP50, mAP50-95
 - PR curves, F1 curves, confusion matrix
 - Per-class metrics
 - Results saved in project/name/
"""

# ---------------- PATH FIX ----------------
import sys, os
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- Imports -----------------
import argparse
from ultralytics import YOLO
import yaml


def load_data_yaml(data_dir: Path):
    """
    Reads or creates a data.yaml for evaluation.
    Returns the path to data.yaml.
    """
    yaml_path = Path(data_dir) / "data.yaml"
    if yaml_path.exists():
        return str(yaml_path)

    # Auto-generate if missing
    names = ["object"]
    data = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)
    return str(yaml_path)


def parse_args():
    p = argparse.ArgumentParser("Validate YOLOv8 Model")
    p.add_argument("--weights", required=True, type=str, help="Path to trained YOLO model (.pt)")
    p.add_argument("--data_dir", type=str, default="data/object_detection_dataset")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="", help="Device id or cpu")
    p.add_argument("--project", type=str, default="outputs/yolo_eval")
    p.add_argument("--name", type=str, default="run_eval")
    p.add_argument("--exist_ok", action="store_true")
    return p.parse_args()


def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    data_yaml = load_data_yaml(data_dir)
    print(f"Using data.yaml: {data_yaml}")

    print(f"Loading YOLO model from: {args.weights}")
    model = YOLO(args.weights)

    print("Running validation...")
    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=(args.device if args.device else None),
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        split="test"
    )

    print("\n=== YOLOv8 Validation Complete ===")
    print("Results saved to:", Path(args.project) / args.name)
    print("Best metrics:")
    try:
        print(f"mAP50:     {results.box.map:.4f}")
        print(f"mAP50-95:  {results.box.map50_95:.4f}")
    except:
        print("⚠️ Could not read mAP results (version difference).")

    return results


if __name__ == "__main__":
    args = parse_args()
    main(args)
