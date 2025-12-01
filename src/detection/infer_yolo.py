"""
src/detection/infer_yolo.py

Run YOLOv8 inference for Bird vs Drone object detection.

This script:
    - Loads a trained YOLOv8 model (.pt file) from models/detection/
    - Runs prediction on:
        * a single image, OR
        * a folder of images
    - Saves visualized results (images with bounding boxes) into:
        results/yolo_infer/<run_name>/

Requirements:
    pip install ultralytics

Assumptions:
    - You have trained a YOLOv8 model using train_yolo.py
    - Best weights are saved in:
        models/detection/yolov8n_best.pt    (or yolov8s_best.pt, etc.)
    - configs/yolov8_data.yaml exists (for auto source mode)

Usage (from project root):

    # 1) Use default model (yolov8n_best.pt) and AUTO test folder from yolov8_data.yaml:
    python -m src.detection.infer_yolo

    # 2) Specify custom model + source image:
    python -m src.detection.infer_yolo --model-path models/detection/yolov8n_best.pt --source "data/object_detection_Dataset/test/image1.jpg"

    # 3) Run on a folder:
    python -m src.detection.infer_yolo --source "data/object_detection_Dataset/test"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

# ultralytics import (with safe fallback)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class YoloInferConfig:
    project_root: Path
    model_path: Path
    source: Path                  # image file or directory
    results_dir: Path             # base directory for saving results

    img_size: Tuple[int, int] = (640, 640)
    device: str = "cpu"           # "cpu" or "0" for GPU
    conf: float = 0.25            # confidence threshold
    run_name: str = "yolov8_infer"  # subfolder inside results_dir


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def get_project_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def resolve_model_path(project_root: Path, model_path_str: str) -> Path:
    p = Path(model_path_str)
    if not p.is_absolute():
        p = project_root / model_path_str
    return p


def resolve_source_path(project_root: Path, source_str: str) -> Path:
    p = Path(source_str)
    if not p.is_absolute():
        p = project_root / source_str
    return p


def has_any_media(path: Path) -> bool:
    """
    Check if a directory contains at least one image/video in supported formats (non-recursive).
    """
    if not path.is_dir():
        return False

    img_exts = {".bmp", ".tif", ".tiff", ".jpeg", ".jpg", ".mpo", ".png", ".dng", ".pfm", ".heic", ".webp"}
    vid_exts = {".mpg", ".avi", ".m4v", ".mkv", ".mov", ".webm", ".gif", ".mpeg", ".wmv", ".ts", ".asf", ".mp4"}

    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in img_exts.union(vid_exts):
            return True
    return False


def parse_yolo_data_yaml(yaml_path: Path) -> Optional[Tuple[str, str]]:
    """
    Very small YAML parser for yolov8_data.yaml created by prepare_yolo.py.

    Returns:
        (dataset_rel_path, test_rel_path) or None if cannot parse.

    Example yaml:
        path: data/object_detection_Dataset
        train: train
        val: val
        test: test
        names:
          0: bird
          1: drone
    """
    if not yaml_path.exists():
        print(f"⚠️ data.yaml not found at: {yaml_path}")
        return None

    dataset_rel = None
    test_rel = None

    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            dataset_rel = line.split("path:", 1)[1].strip()
        elif line.startswith("test:"):
            test_rel = line.split("test:", 1)[1].strip()

    if dataset_rel is None or test_rel is None:
        print(f"⚠️ Could not parse 'path' and 'test' from {yaml_path}")
        return None

    return dataset_rel, test_rel


# ---------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------
def run_inference(cfg: YoloInferConfig) -> None:
    if YOLO is None:
        print("❌ The 'ultralytics' package is not installed.")
        print("   Install it with: pip install ultralytics")
        return

    if not cfg.model_path.exists():
        print(f"❌ Model file not found at: {cfg.model_path}")
        print("   Make sure you've trained YOLO and the path is correct.")
        return

    if not cfg.source.exists():
        print(f"❌ Source path does not exist: {cfg.source}")
        print("   Provide a valid image file or directory, or use --source auto.")
        return

    # If source is a directory, ensure it actually contains images/videos
    if cfg.source.is_dir() and not has_any_media(cfg.source):
        print(f"❌ No images or videos found in: {cfg.source}")
        print("   Check that your test images directory is correct.")
        return

    print("\n📌 YOLOv8 inference configuration:")
    print(asdict(cfg))

    # Create results dir
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print("\n🔧 Loading YOLO model from:", cfg.model_path)
    model = YOLO(str(cfg.model_path))

    # Run predictions
    print("\n🚀 Running inference...")
    results = model.predict(
        source=str(cfg.source),
        imgsz=list(cfg.img_size),
        conf=cfg.conf,
        device=cfg.device,
        save=True,                        # save images with bounding boxes
        project=str(cfg.results_dir),     # base results dir
        name=cfg.run_name,                # subfolder
        exist_ok=True,
        verbose=True,
    )

    # Where outputs are saved
    output_dir = cfg.results_dir / cfg.run_name

    # Basic summary
    num_images = len(results)
    print(f"\n🏁 Inference completed. Processed {num_images} image(s).")
    print(f"✅ Results saved under: {output_dir}")
    print("   (You will find images with bounding boxes there.)")

    # Optional: small text summary of detections
    try:
        import numpy as np  # for convenient counting

        total_detections = 0
        class_counts = {}

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            if boxes.cls is None:
                continue

            cls_ids = boxes.cls.cpu().numpy().astype(int)
            total_detections += len(cls_ids)

            for cid in cls_ids:
                class_counts[cid] = class_counts.get(cid, 0) + 1

        print(f"\n📊 Total detections (all images): {total_detections}")
        if class_counts:
            print("   Detections per class id:")
            for cid, cnt in class_counts.items():
                print(f"    - class {cid}: {cnt}")
        else:
            print("   No objects detected above conf threshold.")
    except Exception:
        # If anything goes wrong, just skip the summary
        pass


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference for Bird vs Drone detection."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/detection/yolov8n_best.pt",
        help="Path to YOLO .pt model (default: models/detection/yolov8n_best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help=(
            "Source image or folder for inference. "
            "If 'auto', will use test path from configs/yolov8_data.yaml. "
            "Otherwise treated as path relative to project root."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Image size as: --img-size H W (default: 640 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or '0' (first GPU), etc. (default: cpu).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/yolo_infer",
        help="Base directory to save inference results (default: results/yolo_infer).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="yolov8_infer",
        help="Subfolder name inside results-dir for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    model_path = resolve_model_path(project_root, args.model_path)

    # Handle 'auto' source: read configs/yolov8_data.yaml
    if args.source == "auto":
        data_yaml = project_root / "configs" / "yolov8_data.yaml"
        parsed = parse_yolo_data_yaml(data_yaml)
        if parsed is None:
            print("❌ Could not auto-detect test images folder from yolov8_data.yaml.")
            print("   Either fix configs/yolov8_data.yaml or pass --source manually.")
            return

        dataset_rel, test_rel = parsed
        source_path = project_root / dataset_rel / test_rel
        print(f"ℹ️ Auto source selected from data.yaml: {source_path}")
    else:
        source_path = resolve_source_path(project_root, args.source)

    results_dir = project_root / args.results_dir

    cfg = YoloInferConfig(
        project_root=project_root,
        model_path=model_path,
        source=source_path,
        results_dir=results_dir,
        img_size=(args.img_size[0], args.img_size[1]),
        device=args.device,
        conf=args.conf,
        run_name=args.run_name,
    )

    run_inference(cfg)


if __name__ == "__main__":
    main()
