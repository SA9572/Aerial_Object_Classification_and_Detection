#!/usr/bin/env python3
"""
train_yolov8.py (robust)
------------------------
Train YOLOv8 object detection (ultralytics API).

Key improvements:
 - Validates dataset layout before calling ulralytics trainer
 - More helpful error messages when folders are missing
 - Allows overriding images/labels subdirectories (for custom layouts)
 - Creates data.yaml only when layout is valid

Usage:
    python src/yolov8/train_yolov8.py \
      --data_dir data/object_detection_dataset \
      --epochs 50 \
      --imgsz 640 \
      --batch 16 \
      --weights yolov8n.pt \
      --project models/yolov8 --name run1

If your dataset uses a different folder layout, pass:
    --images_subdir "imgs" --labels_subdir "annots"
"""

from pathlib import Path
import argparse
import yaml
import sys
import os

# PATH fix so running from anywhere works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ultralytics
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ultralytics is required: pip install ultralytics") from e


def _exists_and_nonempty(p: Path):
    return p.exists() and any(p.glob("*"))


def validate_yolo_layout(root: Path, images_subdir: str, labels_subdir: str):
    """
    Validate that the YOLO dataset layout exists:
      root/images/train, root/images/val, root/images/test
      root/labels/train, root/labels/val, root/labels/test

    Returns: dict with boolean keys and paths
    """
    images_root = root / images_subdir
    labels_root = root / labels_subdir

    required = {}
    for split in ("train", "val", "test"):
        img_dir = images_root / split
        lbl_dir = labels_root / split
        required[f"images_{split}"] = img_dir
        required[f"labels_{split}"] = lbl_dir

    missing = []
    for k, p in required.items():
        if not p.exists() or not any(p.glob("*")):
            missing.append((k, str(p)))

    return required, missing


def create_data_yaml_if_missing(root: Path, yaml_path: Path, images_subdir: str, labels_subdir: str, names=None):
    """
    Create data.yaml at yaml_path using the provided subdirs.
    """
    data = {
        "path": str(root),
        "train": f"{images_subdir}/train",
        "val": f"{images_subdir}/val",
        "test": f"{images_subdir}/test",
        "names": {i: n for i, n in enumerate(names or ["bird", "drone"])}
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(data, open(yaml_path, "w", encoding="utf-8"))
    return str(yaml_path)


def parse_args():
    p = argparse.ArgumentParser("Train YOLOv8 (robust)")
    p.add_argument("--data_dir", type=str, default="data/object_detection_dataset", help="YOLO dataset root")
    p.add_argument("--images_subdir", type=str, default="images", help="images subfolder under data_dir (default: images)")
    p.add_argument("--labels_subdir", type=str, default="labels", help="labels subfolder under data_dir (default: labels)")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="Base weights (yolov8n.pt/yolov8s.pt) or custom")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="models/yolov8", help="project dir to save runs")
    p.add_argument("--name", type=str, default="run1", help="run name")
    p.add_argument("--device", type=str, default="", help="device (e.g. 0 or cpu). Default = auto")
    p.add_argument("--exist_ok", action="store_true", help="allow overwrite run dir")
    p.add_argument("--force_yaml", action="store_true", help="force create data.yaml even if present (overwrites)")
    return p.parse_args()


def main(args):
    data_root = Path(args.data_dir).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}\nCreate it or pass --data_dir pointing to your dataset.")

    # Validate layout
    required_paths, missing = validate_yolo_layout(data_root, args.images_subdir, args.labels_subdir)

    if missing:
        # Nice helpful error with examples and detection attempts
        msg_lines = [
            "ERROR: YOLO dataset layout is incomplete or empty.",
            f"Expected to find the following folders with image/label files (non-empty):"
        ]
        for k, p in missing:
            msg_lines.append(f"  - {k}: {p}")

        msg_lines += [
            "",
            "Ultralytics expects a structure like:",
            "  data/object_detection_dataset/",
            "    images/train/*.jpg",
            "    images/val/*.jpg",
            "    images/test/*.jpg",
            "    labels/train/*.txt",
            "    labels/val/*.txt",
            "    labels/test/*.txt",
            "",
            "If your dataset uses a different layout, re-run with --images_subdir and --labels_subdir",
            "For example, if your images live in 'imgs' and labels in 'annots', run:",
            "  python src/yolov8/train_yolov8.py --data_dir data/object_detection_dataset "
            "--images_subdir imgs --labels_subdir annots",
            "",
            "If you haven't prepared an object-detection dataset yet, convert your dataset to YOLO format first.",
            "See: https://docs.ultralytics.com/datasets"
        ]
        print("\n".join(msg_lines))
        raise SystemExit(1)

    # If data.yaml exists and not forcing overwrite, use it; otherwise create
    yaml_path = data_root / "data.yaml"
    if yaml_path.exists() and not args.force_yaml:
        print("Using existing data.yaml:", yaml_path)
        data_yaml = str(yaml_path)
    else:
        data_yaml = create_data_yaml_if_missing(data_root, yaml_path, args.images_subdir, args.labels_subdir)
        print("Created data.yaml:", data_yaml)

    # Load YOLO model
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
