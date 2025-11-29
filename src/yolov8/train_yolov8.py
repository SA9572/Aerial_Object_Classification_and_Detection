#!/usr/bin/env python3
"""
train_yolov8.py
----------------
Train a YOLOv8 model for aerial object detection (bird/drone).

Expected dataset layout (YOLO format):

data/object_detection_dataset/
    images/
        train/*.jpg|png
        val/*.jpg|png
        test/*.jpg|png
    labels/
        train/*.txt   # YOLO label files
        val/*.txt
        test/*.txt
    data.yaml         # will be created automatically if missing

Run example (from project root):

    python src/yolov8/train_yolov8.py \
        --data_dir data/object_detection_dataset \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --weights yolov8n.pt \
        --project models/yolov8 \
        --name run1
"""

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO


def make_data_yaml_if_missing(data_dir: Path, nc: int = 2, class_names=None) -> Path:
    """
    Ensure data.yaml exists inside data_dir. If not, create a minimal one.
    """
    if class_names is None:
        class_names = ["bird", "drone"]

    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        print(f"Using existing data.yaml: {yaml_path}")
        return yaml_path

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    content = f"""# Auto-generated YOLO dataset config
path: {data_dir.as_posix()}
train: {images_dir.as_posix()}/train
val: {images_dir.as_posix()}/val
test: {images_dir.as_posix()}/test

nc: {nc}
names: {class_names}
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path


def has_any_labels(labels_split_dir: Path) -> bool:
    """
    Return True if there is at least one label file with at least one bbox line.
    """
    if not labels_split_dir.exists():
        return False

    for txt_file in labels_split_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            # at least one line
            return True
    return False


def validate_yolo_dataset(data_dir: Path) -> bool:
    """
    Validate YOLO dataset structure and presence of labels.
    Returns True if OK, False otherwise.
    """
    images_train = data_dir / "images" / "train"
    images_val = data_dir / "images" / "val"
    labels_train = data_dir / "labels" / "train"
    labels_val = data_dir / "labels" / "val"
    images_test = data_dir / "images" / "test"
    labels_test = data_dir / "labels" / "test"

    ok = True

    print(">> Validating YOLO dataset layout...")

    expected = [
        ("images_train", images_train),
        ("images_val", images_val),
        ("labels_train", labels_train),
        ("labels_val", labels_val),
    ]

    for name, p in expected:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            ok = False

    if not ok:
        print("\n❌ YOLO dataset layout is incomplete.")
        print("Expected at least:")
        print(f"  {images_train}")
        print(f"  {labels_train}")
        print(f"  {images_val}")
        print(f"  {labels_val}")
        return False

    # Check at least one label with content for train and val
    if not has_any_labels(labels_train):
        print(f"\n❌ No non-empty label files found in: {labels_train}")
        print("Each .txt file should contain at least one 'class x_center y_center w h' line.")
        ok = False

    if not has_any_labels(labels_val):
        print(f"\n❌ No non-empty label files found in: {labels_val}")
        ok = False

    if not ok:
        print("\nUltralytics will show warnings like 'Labels are missing or empty'.")
        print("Fix: annotate objects in your images or convert from another format to YOLO txt labels.")
        return False

    # Optional test
    if not images_test.exists() or not labels_test.exists():
        print("⚠ WARNING: test split missing. Training/validation will still run.")

    print("✅ YOLO dataset structure looks OK and has some labels.")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for aerial object detection")

    p.add_argument("--data_dir", type=str, default="data/object_detection_dataset",
                   help="Root directory of YOLO dataset (with images/, labels/, data.yaml)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--weights", type=str, default="yolov8n.pt",
                   help="Initial weights (e.g. yolov8n.pt)")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device: 'cpu', '0', '0,1', etc.")
    p.add_argument("--project", type=str, default="models/yolov8",
                   help="Project directory for YOLO runs")
    p.add_argument("--name", type=str, default="run",
                   help="Name of this training run")
    p.add_argument("--exist_ok", action="store_true",
                   help="Allow existing project/name directory")

    return p.parse_args()


def main(args):
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"❌ data_dir not found: {data_dir}")
        sys.exit(1)

    # Validate dataset
    if not validate_yolo_dataset(data_dir):
        sys.exit(1)

    # Ensure data.yaml
    data_yaml = make_data_yaml_if_missing(data_dir, nc=2, class_names=["bird", "drone"])

    print(f"\n>> Starting YOLOv8 training with:")
    print(f"   data    : {data_yaml}")
    print(f"   weights : {args.weights}")
    print(f"   epochs  : {args.epochs}")
    print(f"   imgsz   : {args.imgsz}")
    print(f"   batch   : {args.batch}")
    print(f"   device  : {args.device}")
    print(f"   project : {args.project}")
    print(f"   name    : {args.name}\n")

    model = YOLO(args.weights)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    print("\n✅ YOLOv8 training finished.")
    print(f"Check results under: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main(parse_args())
