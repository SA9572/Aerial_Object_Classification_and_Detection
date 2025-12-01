"""
src/models/utils.py

Utility functions for working with classification models:
    - Seeding
    - Path helpers (project root, models dir)
    - Model file naming / loading
    - Listing available models
    - Plotting training curves (for notebooks)

This module is meant to be imported by:
    - training scripts
    - evaluation scripts
    - notebooks

You can also run it directly to list available models:

    python -m src.models.utils
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------
# Seeding utilities
# ---------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility:
        - Python's random
        - NumPy
        - TensorFlow

    Note:
        Exact reproducibility is not guaranteed across different
        hardware / TF versions, but this helps stabilize results.
    """
    print(f"🌱 Setting global random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Ensuring some deterministic behavior (may slow down training a bit)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def get_project_root(start: Optional[Path] = None) -> Path:
    """
    Try to detect project root.
    - If we're inside 'notebooks', goes one level up.
    - Otherwise, assumes current working directory is root.
    """
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def get_models_dir(project_root: Optional[Path] = None) -> Path:
    """
    Returns the classification models directory:
        <project_root>/models/classification
    """
    if project_root is None:
        project_root = get_project_root()
    models_dir = project_root / "models" / "classification"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# ---------------------------------------------------------------------
# Model filename & loading helpers
# ---------------------------------------------------------------------
def build_model_filename(
    model_type: str,
    backbone: Optional[str] = None,
    suffix: str = "best",
    ext: str = ".h5",
) -> str:
    """
    Build a model filename consistent with train.py:

        model_type="cnn"              -> custom_cnn_best.h5
        model_type="transfer",
        backbone="resnet50"          -> transfer_resnet50_best.h5

    Args:
        model_type: "cnn" or "transfer"
        backbone:  required if model_type="transfer"
        suffix:    string like "best" or "stage1"
        ext:       file extension, default ".h5"
    """
    model_type = model_type.lower()

    if model_type == "cnn":
        name = "custom_cnn"
    elif model_type == "transfer":
        if not backbone:
            raise ValueError("backbone must be provided when model_type='transfer'.")
        name = f"transfer_{backbone.lower()}"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'cnn' or 'transfer'.")

    return f"{name}_{suffix}{ext}"


def get_model_path(
    models_dir: Path,
    model_type: str,
    backbone: Optional[str] = None,
    suffix: str = "best",
) -> Path:
    """
    Return full path to a model file in models_dir.

    Example:
        get_model_path(models_dir, "cnn") ->
            models_dir/custom_cnn_best.h5

        get_model_path(models_dir, "transfer", "resnet50") ->
            models_dir/transfer_resnet50_best.h5
    """
    filename = build_model_filename(
        model_type=model_type,
        backbone=backbone,
        suffix=suffix,
        ext=".h5",
    )
    return models_dir / filename


def load_best_model(
    model_type: str,
    backbone: Optional[str] = None,
    project_root: Optional[Path] = None,
    suffix: str = "best",
) -> tf.keras.Model:
    """
    Load a 'best' model based on model_type + backbone from:
        <project_root>/models/classification

    Args:
        model_type: "cnn" or "transfer"
        backbone:   e.g. "resnet50" if model_type="transfer"
        project_root: base project directory (auto-detected if None)
        suffix:    usually "best" (matching train.py naming)

    Returns:
        Loaded tf.keras.Model

    Raises:
        FileNotFoundError if the model file does not exist.
    """
    if project_root is None:
        project_root = get_project_root()
    models_dir = get_models_dir(project_root)

    model_path = get_model_path(
        models_dir=models_dir,
        model_type=model_type,
        backbone=backbone,
        suffix=suffix,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure you've trained the model and the filename matches."
        )

    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded.")
    return model


def list_available_models(
    project_root: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    List all .h5 model files in <project_root>/models/classification.

    Returns:
        dict with keys 'classification' and a list of filenames.
    """
    if project_root is None:
        project_root = get_project_root()
    models_dir = get_models_dir(project_root)

    if not models_dir.exists():
        return {"classification": []}

    files = sorted(
        [f.name for f in models_dir.glob("*.h5") if f.is_file()]
    )
    return {"classification": files}


# ---------------------------------------------------------------------
# Plotting helpers (for notebooks)
# ---------------------------------------------------------------------
def plot_training_history(
    history: tf.keras.callbacks.History,
    title_prefix: str = "Model",
) -> None:
    """
    Plot training & validation loss/accuracy curves from a History object.

    Usage in notebooks:

        history = model.fit(...)
        from src.models.utils import plot_training_history
        plot_training_history(history, title_prefix="ResNet50")
    """
    hist = history.history

    # Accuracy
    if "accuracy" in hist and "val_accuracy" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(hist["accuracy"], label="train_acc")
        plt.plot(hist["val_accuracy"], label="val_acc")
        plt.title(f"{title_prefix} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Loss
    if "loss" in hist and "val_loss" in hist:
        plt.figure(figsize=(6, 4))
        plt.plot(hist["loss"], label="train_loss")
        plt.plot(hist["val_loss"], label="val_loss")
        plt.title(f"{title_prefix} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility script to inspect available classification models."
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
    print("📁 Project root:", project_root)

    models_dir = get_models_dir(project_root)
    print("📂 Models dir  :", models_dir)

    models = list_available_models(project_root)
    files = models.get("classification", [])

    if not files:
        print("\nℹ️ No .h5 model files found in models/classification yet.")
        print("   Train a model first using: python -m src.models.train")
    else:
        print("\n📦 Available classification models:")
        for name in files:
            print("  -", name)


if __name__ == "__main__":
    main()
