"""
src/deployment/predict.py

Utility functions to run classification (Bird vs Drone) inference
using your trained Keras models.

Default target:
    - Best transfer learning model:
        models/classification/transfer_resnet50_best.h5

Features:
    - Load model and class names (bird, drone)
    - Predict on a single image path
    - CLI interface to test predictions from terminal
    - Reusable functions for Streamlit app or notebooks

Usage from project root (CLI):

    # Use default best transfer ResNet50 model and default classes:
    python -m src.deployment.predict --image "data/classification_dataset/test/bird/xxx.jpg"

    # Specify custom model path:
    python -m src.deployment.predict --image "path/to/image.jpg" --model-path "models/classification/custom_cnn_best.h5"

Usage in Python:

    from src.deployment.predict import load_model_and_classes, predict_image

    model, class_names = load_model_and_classes(
        model_type="transfer",
        backbone="resnet50",
    )
    pred_label, pred_idx, probs = predict_image(
        model,
        "path/to/image.jpg",
        class_names=class_names,
    )
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.utils import load_img, img_to_array

# Reuse some helpers from src.models.utils
from src.models.utils import (
    get_project_root as get_project_root_models,
    load_best_model,
)


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class PredictConfig:
    project_root: Path
    model_type: str = "transfer"        # "cnn" or "transfer"
    backbone: str = "resnet50"          # used when model_type="transfer"
    model_path: Optional[Path] = None   # if None, use load_best_model
    image_size: Tuple[int, int] = (224, 224)
    classes_json_candidates: Tuple[str, ...] = (
        "src/deployment/classes.json",
        "deployment/classes.json",
        "classes.json",
    )


# ---------------------------------------------------------------------
# Helpers: project root, classes
# ---------------------------------------------------------------------
def get_project_root(start: Optional[Path] = None) -> Path:
    """
    Simple wrapper; uses src.models.utils.get_project_root internally.
    """
    return get_project_root_models(start)


def _load_class_names_from_json(project_root: Path, candidates: Tuple[str, ...]) -> Optional[List[str]]:
    """
    Try several possible classes.json locations. Supports either:
        - ["bird", "drone"]
        - {"0": "bird", "1": "drone"}  (string keys)
        - {0: "bird", 1: "drone"}      (int keys)
    """
    for rel in candidates:
        p = project_root / rel
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return list(data)
                elif isinstance(data, dict):
                    # sort by numeric key order
                    # keys may be str or int
                    def _to_int(k):
                        try:
                            return int(k)
                        except Exception:
                            return k

                    items = sorted(data.items(), key=lambda kv: _to_int(kv[0]))
                    return [v for _, v in items]
            except Exception:
                # ignore parse errors and try next candidate
                continue
    return None


def _load_class_names_from_train_folder(project_root: Path) -> Optional[List[str]]:
    """
    Fallback: infer class names from data/classification_dataset/train subfolders.
    """
    train_dir = project_root / "data" / "classification_dataset" / "train"
    if not train_dir.exists():
        return None

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    if not classes:
        return None

    classes.sort()
    return classes


def load_class_names(cfg: PredictConfig) -> List[str]:
    """
    Load class names from:
        1) classes.json (if exists)
        2) train folder (subdirs)
        3) default ["bird", "drone"] as last resort
    """
    # 1) Try JSON
    names = _load_class_names_from_json(cfg.project_root, cfg.classes_json_candidates)
    if names:
        print("✅ Loaded class names from classes.json:", names)
        return names

    # 2) Try train folder
    names = _load_class_names_from_train_folder(cfg.project_root)
    if names:
        print("✅ Inferred class names from train/ folder:", names)
        return names

    # 3) Default
    print("⚠️ Could not load class names from JSON or train folder.")
    print("   Falling back to default: ['bird', 'drone']")
    return ["bird", "drone"]


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model_and_classes(
    model_type: str = "transfer",
    backbone: str = "resnet50",
    model_path: Optional[str | Path] = None,
    image_size: Tuple[int, int] = (224, 224),
    project_root: Optional[Path] = None,
) -> Tuple[tf.keras.Model, List[str], PredictConfig]:
    """
    High-level helper to:
        - Detect project root
        - Load a classification model
        - Load class names

    Args:
        model_type: "cnn" or "transfer"
        backbone:   used when model_type="transfer"
        model_path: optional explicit path to .h5 model
        image_size: (H, W), default (224, 224)
        project_root: optional, auto-detected if None

    Returns:
        model, class_names, cfg
    """
    if project_root is None:
        project_root = get_project_root()

    if isinstance(model_path, str):
        model_path = Path(model_path)

    cfg = PredictConfig(
        project_root=project_root,
        model_type=model_type,
        backbone=backbone,
        model_path=model_path,
        image_size=image_size,
    )

    # Load model
    if cfg.model_path is None:
        print(f"ℹ️ No explicit model_path given. Loading best {model_type} model...")
        model = load_best_model(
            model_type=model_type,
            backbone=backbone if model_type == "transfer" else None,
            project_root=project_root,
            suffix="best",
        )
    else:
        if not cfg.model_path.is_absolute():
            cfg.model_path = project_root / cfg.model_path
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
        print(f"📂 Loading model from: {cfg.model_path}")
        model = keras_load_model(cfg.model_path)
        print("✅ Model loaded.")

    # Load class names
    class_names = load_class_names(cfg)

    return model, class_names, cfg


# ---------------------------------------------------------------------
# Image preprocessing & prediction
# ---------------------------------------------------------------------
def preprocess_image(
    image_path: str | Path,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load a single image and prepare it for model.predict.

    NOTE:
        Your models already include Rescaling layers,
        so we only need to convert to float32 and add batch dimension.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    arr = img_to_array(img)  # shape (H, W, 3), values 0-255
    arr = np.expand_dims(arr, axis=0).astype("float32")  # shape (1, H, W, 3)
    return arr


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: List[str],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[str, int, np.ndarray]:
    """
    Predict class for a single image.

    Returns:
        predicted_label: class name (e.g. "bird")
        predicted_index: int index (e.g. 0)
        probs:           np.array of probabilities (shape: (num_classes,))
    """
    x = preprocess_image(image_path, target_size=target_size)

    preds = model.predict(x, verbose=0)

    # Handle binary vs multi-class outputs
    if preds.shape[-1] == 1:
        # Binary sigmoid output
        prob1 = float(preds[0, 0])
        probs = np.array([1.0 - prob1, prob1], dtype="float32")
    else:
        # Softmax vector
        probs = preds[0].astype("float32")

    pred_idx = int(np.argmax(probs))
    if 0 <= pred_idx < len(class_names):
        pred_label = class_names[pred_idx]
    else:
        pred_label = str(pred_idx)

    return pred_label, pred_idx, probs


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bird vs Drone classification on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image for classification (relative to project root or absolute).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="transfer",
        choices=["cnn", "transfer"],
        help="Model type: 'cnn' or 'transfer' (default: transfer).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone for transfer models (default: resnet50). Ignored for cnn.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit path to .h5 model. If not set, use best model from models/classification/.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size as: --image-size H W (default: 224 224).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = project_root / image_path

    print("📌 Predict configuration:")
    print(
        {
            "project_root": str(project_root),
            "image": str(image_path),
            "model_type": args.model_type,
            "backbone": args.backbone,
            "model_path": args.model_path,
            "image_size": (args.image_size[0], args.image_size[1]),
        }
    )

    # Load model + classes
    model, class_names, cfg = load_model_and_classes(
        model_type=args.model_type,
        backbone=args.backbone,
        model_path=args.model_path,
        image_size=(args.image_size[0], args.image_size[1]),
        project_root=project_root,
    )

    # Predict
    pred_label, pred_idx, probs = predict_image(
        model,
        image_path,
        class_names=class_names,
        target_size=(args.image_size[0], args.image_size[1]),
    )

    print("\n🖼  Image:", image_path)
    print("🔮 Predicted class:", pred_label)
    print("   Class index    :", pred_idx)

    # Pretty print probs
    print("\n📊 Probabilities:")
    for i, p in enumerate(probs):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {i}: {name:>10s} -> {p:.4f}")


if __name__ == "__main__":
    main()
