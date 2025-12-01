"""
src/detection/prepare_yolo.py

Prepare and validate YOLOv8-style object detection dataset
for Bird vs Drone detection.

Supported dataset layouts (relative to project root):

1) SEPARATE IMAGES/LABELS (classic YOLO root-level):

    data/object_detection_Dataset/
        images/
          train/
          val/ or valid/
          test/
        labels/
          train/
          val/ or valid/
          test/

2) SPLIT SUBDIRS (images/labels inside each split):

    data/object_detection_Dataset/
        train/
          images/
          labels/
        val/ or valid/
          images/
          labels/
        test/
          images/
          labels/

3) FLAT SPLITS (images + .txt labels in same folder):

    data/object_detection_Dataset/
        train/
        val/ or valid/
        test/
        # inside each split: *.jpg/*.png + *.txt

This script will:
    - Try to auto-detect which layout you have (with one extra nested level if needed).
    - Validate counts per split (images vs labels).
    - Generate configs/yolov8_data.yaml with correct paths.

Usage (from project root):

    python -m src.detection.prepare_yolo
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class YoloPrepConfig:
    project_root: Path
    dataset_dir: Path           # e.g. <root>/data/object_detection_Dataset
    yaml_out: Path              # e.g. <root>/configs/yolov8_data.yaml
    splits: Tuple[str, ...] = ("train", "val", "test")
    class_names: Tuple[str, ...] = ("bird", "drone")
    layout: str = "unknown"     # "separate_root" | "split_subdirs" | "flat" | "unknown"
    # For layouts:
    # - separate_root: use images_root/split, labels_root/split
    # - split_subdirs: split_dir/images, split_dir/labels
    # - flat:         split_dir (images + labels together)
    images_root: Optional[Path] = None
    labels_root: Optional[Path] = None
    split_dirs: Optional[Dict[str, Path]] = None  # mapping logical split -> actual folder


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


# ---------------------------------------------------------------------
# Utility: list files
# ---------------------------------------------------------------------
def list_files_with_ext(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts = tuple(e.lower() for e in exts)
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


# ---------------------------------------------------------------------
# Check a single split
# ---------------------------------------------------------------------
def check_split(
    images_dir: Path,
    labels_dir: Path,
    split_name: str,
) -> Dict[str, int]:
    if not images_dir.exists():
        print(f"❌ Images folder missing for split '{split_name}': {images_dir}")
        return {"num_images": 0, "num_labels": 0, "missing_labels": 0, "orphan_labels": 0}
    if not labels_dir.exists():
        print(f"❌ Labels folder missing for split '{split_name}': {labels_dir}")
        return {"num_images": 0, "num_labels": 0, "missing_labels": 0, "orphan_labels": 0}

    image_files = list_files_with_ext(images_dir, (".jpg", ".jpeg", ".png", ".bmp"))
    label_files = list_files_with_ext(labels_dir, (".txt",))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    num_images = len(image_files)
    num_labels = len(label_files)
    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print(f"\n📂 Split: {split_name}")
    print(f"  Images: {num_images}")
    print(f"  Labels: {num_labels}")

    if missing_labels:
        print(f"  ⚠️ Missing label files for {len(missing_labels)} images (no .txt).")
        for s in list(missing_labels)[:5]:
            print(f"    - {s}.jpg (or .png)")
    else:
        print("  ✅ Every image seems to have a label file.")

    if orphan_labels:
        print(f"  ⚠️ Orphan label files without matching images: {len(orphan_labels)}")
        for s in list(orphan_labels)[:5]:
            print(f"    - {s}.txt")
    else:
        print("  ✅ No orphan label files detected.")

    return {
        "num_images": num_images,
        "num_labels": num_labels,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
    }


# ---------------------------------------------------------------------
# Layout detection helpers
# ---------------------------------------------------------------------
SPLIT_SYNONYMS = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing", "eval"],
}


def find_split_dirs(dataset_dir: Path) -> Dict[str, Path]:
    """
    Find actual split directories under dataset_dir using synonyms.
    Returns mapping logical_split -> actual Path if found.
    """
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    name_to_dir = {d.name.lower(): d for d in subdirs}

    split_dirs: Dict[str, Path] = {}
    for logical, synonyms in SPLIT_SYNONYMS.items():
        for syn in synonyms:
            if syn in name_to_dir:
                split_dirs[logical] = name_to_dir[syn]
                break

    return split_dirs


def detect_layout(dataset_dir: Path) -> YoloPrepConfig:
    """
    Try to detect dataset layout. Also handles one extra nesting level
    (e.g., data/object_detection_Dataset/object_detection_Dataset/...).
    Returns a YoloPrepConfig with layout + paths filled (except yaml_out).
    """
    project_root = get_project_root()
    cfg = YoloPrepConfig(
        project_root=project_root,
        dataset_dir=dataset_dir,
        yaml_out=project_root / "configs" / "yolov8_data.yaml",
        layout="unknown",
    )

    if not dataset_dir.exists():
        print(f"❌ Dataset dir does not exist: {dataset_dir}")
        return cfg

    # 1) Try root-level separate layout: images/ + labels/
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if images_root.exists() and labels_root.exists():
        print("🔍 Detected layout: SEPARATE ROOT (images/ + labels/)")
        cfg.layout = "separate_root"
        cfg.images_root = images_root
        cfg.labels_root = labels_root
        return cfg

    # 2) Try split subdirs or flat under dataset_dir
    split_dirs = find_split_dirs(dataset_dir)
    if split_dirs:
        # Check if inside each split we have images/ + labels/
        has_split_subdirs = all(
            (split_dirs.get(k) / "images").exists() and (split_dirs.get(k) / "labels").exists()
            for k in ["train", "val"]
            if k in split_dirs
        )

        if has_split_subdirs:
            print("🔍 Detected layout: SPLIT SUBDIRS (train/images, train/labels, ...)")
            cfg.layout = "split_subdirs"
            cfg.split_dirs = split_dirs
            return cfg
        else:
            print("🔍 Detected layout: FLAT SPLITS (train/, val/, test/ with images + txt)")
            cfg.layout = "flat"
            cfg.split_dirs = split_dirs
            return cfg

    # 3) If no splits found, but there's exactly one subdir, descend one level and retry
    children = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if len(children) == 1:
        nested = children[0]
        print(f"ℹ️ No splits found directly under {dataset_dir}, trying nested dir: {nested}")
        return detect_layout(nested)

    # 4) Give up gracefully
    print("❌ Could not detect YOLO dataset layout under:", dataset_dir)
    print("   Expected either:")
    print("    - images/ + labels/ at root")
    print("    - train/, val/ (or valid/), test/ under dataset dir")
    cfg.layout = "unknown"
    return cfg


# ---------------------------------------------------------------------
# Inspect dataset by layout
# ---------------------------------------------------------------------
def inspect_dataset(cfg: YoloPrepConfig) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Inspect dataset splits according to detected layout.
    Returns summary dict or None if layout unknown.
    """
    print("📁 YOLO dataset dir  :", cfg.dataset_dir)
    print("Layout               :", cfg.layout)
    print("Classes              :", cfg.class_names)

    if cfg.layout == "unknown":
        print("\n⚠️ Layout is unknown. Skipping inspection.")
        return None

    summary: Dict[str, Dict[str, int]] = {}

    if cfg.layout == "separate_root":
        if cfg.images_root is None or cfg.labels_root is None:
            print("⚠️ images_root or labels_root not set. Skipping inspection.")
            return None
        for logical_split in ["train", "val", "test"]:
            images_dir = cfg.images_root / logical_split
            labels_dir = cfg.labels_root / logical_split
            res = check_split(images_dir, labels_dir, logical_split)
            summary[logical_split] = res

    elif cfg.layout in ("split_subdirs", "flat"):
        if not cfg.split_dirs:
            print("⚠️ split_dirs mapping not set. Skipping inspection.")
            return None
        for logical_split in ["train", "val", "test"]:
            if logical_split not in cfg.split_dirs:
                print(f"ℹ️ No '{logical_split}' split directory found. Skipping.")
                continue
            split_dir = cfg.split_dirs[logical_split]

            if cfg.layout == "split_subdirs":
                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"
            else:  # flat
                images_dir = split_dir
                labels_dir = split_dir

            res = check_split(images_dir, labels_dir, logical_split)
            summary[logical_split] = res

    print("\n📊 Summary:")
    for split_name, res in summary.items():
        print(
            f"  {split_name}: images={res['num_images']}, labels={res['num_labels']}, "
            f"missing_labels={res['missing_labels']}, orphan_labels={res['orphan_labels']}"
        )

    return summary


# ---------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------
def generate_yolov8_yaml(cfg: YoloPrepConfig) -> None:
    """
    Generate yolov8_data.yaml based on detected layout.

    For SEPARATE ROOT:
        path: data/object_detection_Dataset
        train: images/train
        val: images/val or images/valid
        test: images/test

    For SPLIT SUBDIRS:
        path: data/object_detection_Dataset
        train: train/images
        val: val/images
        test: test/images

    For FLAT SPLITS:
        path: data/object_detection_Dataset
        train: train
        val: val
        test: test
    """
    if cfg.layout == "unknown":
        print("\n⚠️ Layout unknown. Not generating yolov8_data.yaml.")
        return

    dataset_name = cfg.dataset_dir.name
    dataset_rel_path = f"data/{dataset_name}"

    lines = [f"path: {dataset_rel_path}"]

    if cfg.layout == "separate_root":
        # standard names: train, val, test
        lines.append("train: images/train")
        # val split may be "val" or "valid" in file system, but YAML just wants path
        lines.append("val: images/val")
        lines.append("test: images/test")

    elif cfg.layout == "split_subdirs":
        # here split_dirs contain actual folder names (train/valid/test)
        if not cfg.split_dirs:
            print("⚠️ split_dirs not set. Cannot generate YAML.")
            return

        train_dir_name = cfg.split_dirs["train"].name if "train" in cfg.split_dirs else "train"
        val_dir_name = cfg.split_dirs["val"].name if "val" in cfg.split_dirs else "val"
        test_dir_name = cfg.split_dirs["test"].name if "test" in cfg.split_dirs else "test"

        lines.append(f"train: {train_dir_name}/images")
        lines.append(f"val: {val_dir_name}/images")
        lines.append(f"test: {test_dir_name}/images")

    elif cfg.layout == "flat":
        if not cfg.split_dirs:
            print("⚠️ split_dirs not set. Cannot generate YAML.")
            return

        train_dir_name = cfg.split_dirs["train"].name if "train" in cfg.split_dirs else "train"
        val_dir_name = cfg.split_dirs["val"].name if "val" in cfg.split_dirs else "val"
        test_dir_name = cfg.split_dirs["test"].name if "test" in cfg.split_dirs else "test"

        lines.append(f"train: {train_dir_name}")
        lines.append(f"val: {val_dir_name}")
        lines.append(f"test: {test_dir_name}")

    lines.append("names:")
    for idx, cls_name in enumerate(cfg.class_names):
        lines.append(f"  {idx}: {cls_name}")

    yaml_text = "\n".join(lines) + "\n"

    cfg.yaml_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.yaml_out.write_text(yaml_text, encoding="utf-8")

    print("\n📝 YOLOv8 data.yaml generated at:", cfg.yaml_out)
    print("\n--- yolov8_data.yaml content ---")
    print(yaml_text)
    print("--------------------------------")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate YOLOv8 object detection dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="object_detection_Dataset",
        help="Relative directory under data/ for YOLO dataset (default: object_detection_Dataset).",
    )
    parser.add_argument(
        "--yaml-out",
        type=str,
        default="configs/yolov8_data.yaml",
        help="Output path for YOLOv8 data.yaml (default: configs/yolov8_data.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    dataset_dir = project_root / "data" / args.dataset_dir
    yaml_out = project_root / args.yaml_out

    cfg = detect_layout(dataset_dir)
    cfg.project_root = project_root
    cfg.yaml_out = yaml_out

    print("\n📌 YOLO preparation config:")
    print(asdict(cfg))

    # If layout couldn't be detected, just exit gracefully.
    if cfg.layout == "unknown":
        print("\n❌ Could not detect a valid YOLO dataset layout. Please check your")
        print("   folder structure under:", dataset_dir)
        return

    # 1) Inspect dataset
    inspect_dataset(cfg)

    # 2) Generate YAML
    generate_yolov8_yaml(cfg)

    print("\n✅ YOLO dataset preparation completed.")


if __name__ == "__main__":
    main()
