#!/usr/bin/env python3
"""
train_transfer_learning.py
--------------------------
Training script for the Transfer Learning model (EfficientNetB0 default).

- Loads classification dataset from: data/classification_dataset
- Builds a transfer learning model using src.models.transfer_learing.build_transfer_model
- Optionally fine-tunes the top N backbone layers if --fine_tune > 0
- Saves:
    - best_model.h5 (from callbacks)
    - history .pkl and .json

Run from project root, e.g.:

    python src/training/train_transfer_learning.py \
        --data_dir data/classification_dataset \
        --out_dir models/transfer_learning \
        --img_size 224 224 \
        --batch_size 32 \
        --epochs 20 \
        --lr 1e-4 \
        --backbone EfficientNetB0 \
        --fine_tune 0

"""

# -------------------------------
# PATH FIX: make 'src' importable
# -------------------------------
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

src_dir = Path(PROJECT_ROOT) / "src"
models_dir = src_dir / "models"
src_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# ensure packages
init_src = src_dir / "__init__.py"
if not init_src.exists():
    init_src.write_text("", encoding="utf-8")
init_models = models_dir / "__init__.py"
if not init_models.exists():
    init_models.write_text("", encoding="utf-8")

# -------------------------------
# Imports
# -------------------------------
import argparse
import tensorflow as tf

from src.data.dataloader import DataLoader
from src.models.transfer_learing import build_transfer_model, unfreeze_backbone_top_layers
from src.training.callbacks import get_all_callbacks
from src.models.save_load import save_history, ensure_dir


# -------------------------------
# Argument parsing
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Transfer Learning Model for Aerial Classification")

    p.add_argument(
        "--data_dir",
        type=str,
        default="data/classification_dataset",
        help="Path to classification dataset with train/valid/test folders",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="models/transfer_learning",
        help="Directory to save model, checkpoints and logs",
    )
    p.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size H W (default 224 224)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="EfficientNetB0",
        choices=["EfficientNetB0", "MobileNetV2", "ResNet50"],
        help="Backbone for transfer learning",
    )
    p.add_argument(
        "--fine_tune",
        type=int,
        default=0,
        help="If >0, number of top backbone layers to unfreeze for fine-tuning",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Enable TensorBoard logging in callbacks",
    )
    p.add_argument(
        "--monitor",
        type=str,
        default="val_accuracy",
        help="Metric name monitored by ModelCheckpoint / EarlyStopping callbacks",
    )
    return p.parse_args()


# -------------------------------
# Main training logic
# -------------------------------
def main(args):
    out_dir = ensure_dir(args.out_dir)

    print(
        f"\n===== Transfer Learning Training =====\n"
        f"Data Dir      : {args.data_dir}\n"
        f"Output Dir    : {out_dir}\n"
        f"Image Size    : {tuple(args.img_size)}\n"
        f"Batch Size    : {args.batch_size}\n"
        f"Epochs        : {args.epochs}\n"
        f"Learning Rate : {args.lr}\n"
        f"Backbone      : {args.backbone}\n"
        f"Fine-tune     : Top {args.fine_tune} layers\n"
    )

    # Detect GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs Detected : {[g.name for g in gpus]}")
    else:
        print("⚠ No GPU detected (training may be slow).")

    # ---------------- Load dataset ----------------
    loader = DataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=True,  # IMPORTANT: images normalized to [0,1]
    )

    train_ds, val_ds, test_ds, class_names = loader.load()
    num_classes = len(class_names)

    print(f"📌 Classes found: {class_names}")

    # ---------------- Build model ----------------
    # Images are already normalized to [0,1] by DataLoader.
    # Our transfer_learing.build_transfer_model expects [0,1] inputs
    model = build_transfer_model(
        input_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=num_classes,
        backbone=args.backbone,
        base_trainable=False,     # backbone frozen initially
        dropout_rate=0.3,
        lr=args.lr,
        metrics=None,             # will use default metrics in builder
    )

    model.summary()

    # ---------------- Optional fine-tuning ----------------
    if args.fine_tune > 0:
        print(f"\nUnfreezing top {args.fine_tune} layers of backbone for fine-tuning...")
        unfreeze_backbone_top_layers(model, unfreeze_layers=args.fine_tune)
        print("✓ Layers marked trainable. Recompiling model with smaller LR for fine-tuning...")
        fine_tune_lr = args.lr * 0.1
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=model.loss,
            metrics=model.metrics,
        )
        print(f"Fine-tune learning rate: {fine_tune_lr}\n")

    # ---------------- Callbacks ----------------
    callbacks = get_all_callbacks(
        out_dir=str(out_dir),
        use_tensorboard=args.use_tensorboard,
        monitor=args.monitor,
    )

    # ---------------- Training ----------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ---------------- Save history ----------------
    hist_path = Path(out_dir) / "history"
    pkl_path, json_path = save_history(history, str(hist_path))
    print(f"\nHistory saved to:\n  {pkl_path}\n  {json_path}")

    # ---------------- Evaluate on test set ----------------
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test Results:", results)

    # ---------------- Save final model if no best checkpoint ----------------
    best_model_path = Path(out_dir) / "best_model.h5"
    if best_model_path.exists():
        print(f"\nBest model saved at:\n  {best_model_path}")
    else:
        fallback = Path(out_dir) / "final_model.h5"
        print("⚠ No checkpoint found, saving final model manually...")
        model.save(str(fallback))
        print(f"Final model saved at:\n  {fallback}")

    print("\n===== Training Complete =====\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
