#!/usr/bin/env python3
"""
prepare_yolo_dataset.py

Create YOLOv8 dataset structure and split images into train/val/test.

Usage examples (run from project root):
  # Basic: source images in data/images_all, optional labels in data/labels_all
  python scripts/prepare_yolo_dataset.py \
    --images_src data/images_all \
    --labels_src data/labels_all \
    --out data/object_detection_dataset \
    --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

Notes:
 - If labels_src is provided, the script will look for .txt files with the same stem as each image
   and move them to the corresponding labels/<split>/ folder.
 - If labels_src is NOT provided, the script will create empty .txt label files for each image.
   You should annotate them properly before training.
"""

import argparse
import random
from pathlib import Path
import shutil
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_src", required=True, help="Source folder containing all images")
    p.add_argument("--labels_src", default=None, help="Source folder containing YOLO .txt labels (optional)")
    p.add_argument("--out", default="data/object_detection_dataset", help="Output YOLO dataset root")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png"])
    return p.parse_args()

def main(args):
    images_src = Path(args.images_src).resolve()
    if not images_src.exists():
        print("ERROR: images_src not found:", images_src)
        sys.exit(1)

    labels_src = Path(args.labels_src).resolve() if args.labels_src else None
    if labels_src and not labels_src.exists():
        print("ERROR: labels_src not found:", labels_src)
        sys.exit(1)

    out_root = Path(args.out).resolve()
    images_out = out_root / "images"
    labels_out = out_root / "labels"

    # Make target directories
    for d in [
        images_out / "train", images_out / "val", images_out / "test",
        labels_out / "train", labels_out / "val", labels_out / "test"
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Gather image files
    img_files = [p for ext in args.extensions for p in images_src.rglob(f"*{ext}")]
    img_files = sorted(set(img_files))
    if not img_files:
        print("No images found in", images_src, "with extensions", args.extensions)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(img_files)

    n = len(img_files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train+n_val]
    test_files = img_files[n_train+n_val:]

    splits = [("train", train_files), ("val", val_files), ("test", test_files)]
    print(f"Total images: {n} -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    for split_name, files in splits:
        for src_img in files:
            dst_img = images_out / split_name / src_img.name
            shutil.copy2(src_img, dst_img)

            # corresponding label
            label_name = src_img.with_suffix(".txt").name
            if labels_src:
                src_label = labels_src / label_name
                if src_label.exists():
                    shutil.copy2(src_label, labels_out / split_name / label_name)
                else:
                    # create empty label if not found
                    (labels_out / split_name / label_name).write_text("", encoding="utf-8")
            else:
                # make empty label placeholder
                (labels_out / split_name / label_name).write_text("", encoding="utf-8")

    print("Dataset prepared at:", out_root)
    print("images/:", [p for p in (images_out).iterdir() if p.is_dir()])
    print("labels/:", [p for p in (labels_out).iterdir() if p.is_dir()])
    print("IMPORTANT: If you created empty .txt label files, annotate them properly before training.")
    print("Now you can run your train_yolov8.py with --data_dir", out_root)

if __name__ == "__main__":
    args = parse_args()
    main(args)
