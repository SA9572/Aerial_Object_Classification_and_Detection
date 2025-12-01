"""
src/models/train.py

Training script for Bird vs Drone classification.

Features:
    - Train either:
        * Custom CNN           (model-type: "cnn")
        * Transfer Learning    (model-type: "transfer" + backbone)
          - resnet50
          - mobilenetv2
          - efficientnetb0

    - Uses folder-based dataset:
        data/classification_dataset/
          train/
            bird/
            drone/
          valid/
            bird/
            drone/

    - Uses tf.data.image_dataset_from_directory (memory-safe)
    - Saves best model to models/classification/
    - Optionally logs training history to logs/training_logs.csv

Run from project root, for example:

    # Transfer learning with ResNet50
    python -m src.models.train --model-type transfer --backbone resnet50

    # Custom CNN
    python -m src.models.train --model-type cnn
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import model builders
from src.models.cnn_model import build_custom_cnn
from src.models.transfer_model import build_transfer_model


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_type: str = "transfer"          # "cnn" or "transfer"
    backbone: str = "resnet50"            # for transfer: "resnet50"|"mobilenetv2"|"efficientnetb0"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 20
    train_base: bool = False              # for transfer: fine-tune backbone or not
    pretrained_weights: str | None = None # leave None for low-RAM machines
    # Paths will be filled at runtime:
    project_root: Path | None = None
    data_root: Path | None = None
    models_dir: Path | None = None
    logs_dir: Path | None = None


# ---------------------------------------------------------------------
# Helper: project root & dataset paths
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def get_dataset_paths(project_root: Path) -> Tuple[Path, Path]:
    """
    Returns train_dir, val_dir inside data/classification_dataset.
    """
    base = project_root / "data" / "classification_dataset"
    train_dir = base / "train"
    val_dir = base / "valid"
    return train_dir, val_dir


# ---------------------------------------------------------------------
# Helper: create tf.data datasets
# ---------------------------------------------------------------------
def create_datasets(
    train_dir: Path,
    val_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Create train and validation tf.data datasets from folders.
    """
    print("Creating datasets from:")
    print("  Train:", train_dir)
    print("  Valid:", val_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Valid directory not found: {val_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",  # integer labels
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    print("\nClass names:", class_names)
    print("num_classes:", len(class_names))

    # Optimize input pipeline (no cache to keep RAM usage low)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


# ---------------------------------------------------------------------
# Helper: build model based on config
# ---------------------------------------------------------------------
def build_model_from_config(
    cfg: TrainConfig,
    num_classes: int,
) -> tf.keras.Model:
    input_shape = (cfg.image_size[0], cfg.image_size[1], 3)

    if cfg.model_type == "cnn":
        print("\n🔧 Building Custom CNN model...")
        model = build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
        )
        model_name = "custom_cnn"

    elif cfg.model_type == "transfer":
        print(f"\n🔧 Building Transfer model (backbone={cfg.backbone})...")
        model = build_transfer_model(
            backbone=cfg.backbone,
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=0.3,
            train_base=cfg.train_base,
            pretrained_weights=cfg.pretrained_weights,  # default None for safety
        )
        model_name = f"transfer_{cfg.backbone.lower()}"

    else:
        raise ValueError(f"Unknown model_type '{cfg.model_type}'. Use 'cnn' or 'transfer'.")

    model._model_name_for_saving = model_name  # custom attribute for convenience
    return model


# ---------------------------------------------------------------------
# Helper: compile model
# ---------------------------------------------------------------------
def compile_model(model: tf.keras.Model) -> None:
    """
    Compile model with a standard setup for classification.
    We use sparse_categorical_crossentropy because labels are integer-encoded.
    """
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )


# ---------------------------------------------------------------------
# Helper: training callbacks
# ---------------------------------------------------------------------
def get_callbacks(
    cfg: TrainConfig,
    model: tf.keras.Model,
) -> list[tf.keras.callbacks.Callback]:
    """
    Create callbacks:
        - EarlyStopping (patience=5 on val_loss)
        - ModelCheckpoint (save best model)
    """
    # Determine filename based on model type/backbone
    model_name = getattr(model, "_model_name_for_saving", "model")
    best_model_path = cfg.models_dir / f"{model_name}_best.h5"

    print("\n📁 Best model will be saved to:", best_model_path)

    ckpt_cb = ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    return [ckpt_cb, early_cb]


# ---------------------------------------------------------------------
# Helper: log training history
# ---------------------------------------------------------------------
def log_history(
    cfg: TrainConfig,
    history: tf.keras.callbacks.History,
) -> None:
    """
    Append training history to logs/training_logs.csv
    (one row per epoch with some basic info).
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.logs_dir / "training_logs.csv"

    fieldnames = [
        "timestamp",
        "model_type",
        "backbone",
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
    ]

    rows = []
    hist = history.history
    epochs = len(hist.get("loss", []))
    timestamp = datetime.now().isoformat(timespec="seconds")

    for epoch in range(epochs):
        row = {
            "timestamp": timestamp,
            "model_type": cfg.model_type,
            "backbone": cfg.backbone if cfg.model_type == "transfer" else "",
            "epoch": epoch + 1,
            "loss": float(hist["loss"][epoch]),
            "accuracy": float(hist["accuracy"][epoch]),
            "val_loss": float(hist["val_loss"][epoch]),
            "val_accuracy": float(hist["val_accuracy"][epoch]),
        }
        rows.append(row)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print("\n📝 Training history appended to:", csv_path)


# ---------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Bird vs Drone classification model."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "transfer"],
        default="transfer",
        help="Which model to train: 'cnn' or 'transfer' (default: transfer).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet50", "mobilenetv2", "efficientnetb0"],
        default="resnet50",
        help="Backbone for transfer learning (if model-type=transfer).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size as: --image-size H W (default: 224 224).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--train-base",
        action="store_true",
        help="If set and model-type=transfer, fine-tune backbone (base model trainable).",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help="Pretrained weights to use (default: None). Use 'imagenet' ONLY if you have enough RAM.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Build configuration object
    cfg = TrainConfig(
        model_type=args.model_type,
        backbone=args.backbone,
        image_size=(args.image_size[0], args.image_size[1]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_base=args.train_base,
        pretrained_weights=args.pretrained_weights,
    )

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    cfg.project_root = project_root
    cfg.data_root = project_root / "data"
    cfg.models_dir = project_root / "models" / "classification"
    cfg.logs_dir = project_root / "logs"

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    print("📌 Training configuration:")
    print(asdict(cfg))

    # Dataset paths
    train_dir, val_dir = get_dataset_paths(project_root)

    # Create datasets
    train_ds, val_ds, class_names = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )
    num_classes = len(class_names)

    # Build model
    model = build_model_from_config(cfg, num_classes=num_classes)
    print("\nModel summary:\n")
    model.summary()

    # Compile
    compile_model(model)

    # Callbacks
    callbacks = get_callbacks(cfg, model)

    # Train
    print("\n🚀 Starting training...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Log history
    log_history(cfg, history)

    print("\n✅ Training finished.")
    print("You can now use the best model for inference or Streamlit deployment.")


if __name__ == "__main__":
    main()
