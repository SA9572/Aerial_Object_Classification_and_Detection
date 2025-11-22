#!/usr/bin/env python3
"""
infer_yolov8.py
---------------
Run inference with a YOLOv8 model on an image, folder of images, video, or webcam.

Usage examples:
  # Single image
  python src/yolov8/infer_yolov8.py --weights models/yolov8/run1/weights/best.pt --source data/images/test/img1.jpg

  # Folder of images
  python src/yolov8/infer_yolov8.py --weights models/yolov8/run1/weights/best.pt --source data/images/test --save_txt

  # Video file (will produce annotated video)
  python src/yolov8/infer_yolov8.py --weights models/yolov8/run1/weights/best.pt --source videos/input.mp4

  # Webcam (use integer index)
  python src/yolov8/infer_yolov8.py --weights yolov8n.pt --source 0 --device 0

Notes:
 - Requires `pip install ultralytics`
 - The script will save outputs to --project/--name (default: outputs/yolo_infer/run)
 - For a single image the annotated image will be saved; for directory all images will be processed.
 - If you don't have a trained model yet, you may use a pretrained weight such as yolov8n.pt (will be downloaded automatically).
"""

import argparse
from pathlib import Path
import sys
import os

# Optional: example of an uploaded file path in this environment (you do not need it for inference)
# UPLOADED_FILE = "/mnt/data/Project Title.pdf"

# PATH fix so script works when executed from anywhere in project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ultralytics is required. Install with: pip install ultralytics") from e


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 inference (image/folder/video/webcam)")
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt) or model name (yolov8n.pt)")
    p.add_argument("--source", type=str, required=True,
                   help="Source: image file, folder of images, video file, or webcam index (0,1...). "
                        "Examples: data/images/test.jpg, data/images/test_dir, video.mp4, 0")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", type=str, default="", help="Device, e.g. '0' or 'cpu' (default auto)")
    p.add_argument("--save_dir", type=str, default="outputs/yolo_infer", help="Directory to save results")
    p.add_argument("--name", type=str, default="run", help="Run name subfolder inside save_dir")
    p.add_argument("--save_txt", action="store_true", help="Save detection results as YOLO-format .txt files")
    p.add_argument("--save_conf", action="store_true", help="Save confidences in labels.txt (ultralytics save_conf)")
    p.add_argument("--show", action="store_true", help="Display results in a window (may not work on headless servers)")
    p.add_argument("--classes", type=int, nargs="+", default=None, help="Filter by class ids (e.g. 0 1)")
    return p.parse_args()


def prepare_source_arg(source: str):
    """
    Prepare the source argument for ultralytics:
    - If numeric (webcam), return as-is
    - If file/folder, ensure it exists; if not, raise
    """
    # Webcam (integer)
    try:
        idx = int(source)
        return idx  # ultralytics accepts int for webcam
    except Exception:
        pass

    p = Path(source)
    if p.exists():
        return str(p)
    else:
        # try to be helpful: maybe user passed relative path without ./ or backslash issues
        alt = Path.cwd() / source
        if alt.exists():
            return str(alt)
        raise FileNotFoundError(f"Source not found: {source}")


def ensure_out_dir(save_dir: str, name: str):
    out_dir = Path(save_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_inference(args):
    # Validate source
    try:
        src = prepare_source_arg(args.source)
    except FileNotFoundError as e:
        print("ERROR:", e)
        sys.exit(1)

    out_dir = ensure_out_dir(args.save_dir, args.name)
    print(f"Saving inference outputs to: {out_dir}")

    # Load model
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"ERROR loading model weights '{args.weights}': {e}")
        sys.exit(1)

    predict_kwargs = dict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=(args.device if args.device else None),
        save=args.save_txt is False and not args.show,  # leave saving to ultralytics if not using save_txt/shows
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(out_dir.parent),  # ultralytics will create project/name
        name=str(out_dir.name),
        exist_ok=True,
        show=args.show,
    )

    # If user requested class filter, pass classes
    if args.classes is not None:
        predict_kwargs["classes"] = args.classes

    # Important: ultralytics model.predict returns a list of Results objects.
    print("Running inference with the following parameters:")
    print(f"  source: {src}")
    print(f"  imgsz: {args.imgsz}, conf: {args.conf}, iou: {args.iou}, device: {args.device or 'auto'}")
    try:
        results = model.predict(**predict_kwargs)
    except Exception as e:
        # Try a fallback call signature for older ultralytics versions
        try:
            results = model.predict(source=src, imgsz=args.imgsz, conf=args.conf, device=(args.device or None),
                                    save=args.save_txt is False and not args.show, save_txt=args.save_txt,
                                    project=str(out_dir.parent), name=str(out_dir.name), exist_ok=True)
        except Exception as e2:
            print("ERROR during model.predict():", e2)
            sys.exit(1)

    # Summarize results
    # Ultraytics saves outputs under project/name by default.
    saved_location = out_dir
    print("Inference finished. Check outputs in:", saved_location)

    # print a short summary: number of images/frames processed and detections
    total_images = 0
    total_detections = 0
    for res in results:
        try:
            # res.orig_shape may exist; res.boxes or res.boxes.data contains detections
            n = len(res.boxes) if hasattr(res, "boxes") else (len(res.masks) if hasattr(res, "masks") else 0)
            total_detections += n
        except Exception:
            pass
        total_images += 1

    print(f"Processed {total_images} inputs; total detections across results (approx): {total_detections}")
    print("If you set --save_txt the YOLO-format .txt files will be in the same output folder.")

    return results, saved_location


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
