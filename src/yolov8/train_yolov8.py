#!/usr/bin/env python3
"""
train_yolov8.py
---------------
Train YOLOv8 object detection (ultralytics API).

Usage (example):
    python src/yolov8/train_yolov8.py \
      --data_dir data/object_detection_dataset \
      --epochs 50 \
      --imgsz 640 \
      --batch 16 \
      --weights yolov8n.pt \
      --project models/yolov8 --name run1

Notes:
 - Requires `pip install ultralytics`
 - Expects YOLO-format dataset:
     data/object_detection_dataset/
         images/train/*.jpg
         images/val/*.jpg
         images/test/*.jpg
         labels/train/*.txt
         labels/val/*.txt
         labels/test/*.txt
 - Creates data.yaml automatically if not present.
"""

# PATH fix
import sys, os
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import yaml

# ultralytics
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ultralytics is required: pip install ultralytics") from e


def create_data_yaml(od_dir: Path, names=None):
    """
    Create a minimal data.yaml for YOLOv8 if not present.
    """
    od_dir = Path(od_dir)
    yaml_path = od_dir / "data.yaml"
    if yaml_path.exists():
        return str(yaml_path)

    names = names or ["bird", "drone"]
    data = {
        "path": str(od_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)}
    }
    od_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return str(yaml_path)


def parse_args():
    p = argparse.ArgumentParser("Train YOLOv8")
    p.add_argument("--data_dir", type=str, default="data/object_detection_dataset", help="YOLO dataset root")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="Base weights (yolov8n.pt/yolov8s.pt) or custom")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="models/yolov8", help="project dir to save runs")
    p.add_argument("--name", type=str, default="run1", help="run name")
    p.add_argument("--device", type=str, default="", help="device (e.g. 0 or cpu). Default = auto")
    p.add_argument("--exist_ok", action="store_true", help="allow overwrite run dir")
    return p.parse_args()


def main(args):
    od_dir = Path(args.data_dir)
    if not od_dir.exists():
        raise FileNotFoundError(f"Object detection dataset directory not found: {od_dir}")

    data_yaml = create_data_yaml(od_dir)
    print("Using data.yaml:", data_yaml)

    # Build YOLO model wrapper using specified weights
    model = YOLO(args.weights)

    print(f"Starting training: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        device=(args.device if args.device else None),
        exist_ok=args.exist_ok
    )
    print("YOLOv8 training complete. Check results in:", Path(args.project) / args.name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
