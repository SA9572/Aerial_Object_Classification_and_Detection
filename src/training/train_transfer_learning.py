# #!/usr/bin/env python3
# """
# train_transfer_learning.py
# --------------------------
# Training script for the Transfer Learning model (EfficientNetB0 default).

# This version:
#  - ensures project root is on sys.path
#  - auto-creates __init__.py in src/ and src/models/ if missing so imports like
#    `from src.models.transfer_learning import ...` work reliably on all platforms
# """

# # -------------------------------
# # PATH FIX + ensure package __init__.py files
# # -------------------------------
# import sys
# import os
# from pathlib import Path

# # Add project root (two levels up) to sys.path so 'src' is importable
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # Ensure src/ and src/models/ are proper packages (create __init__.py if needed)
# src_dir = Path(PROJECT_ROOT) / "src"
# models_dir = src_dir / "models"

# try:
#     src_dir.mkdir(parents=False, exist_ok=True)
#     models_dir.mkdir(parents=True, exist_ok=True)
# except Exception:
#     # If creation fails, continue — imports may still work if directories already existed
#     pass

# # Create __init__.py files if they don't exist
# if not (src_dir / "__init__.py").exists():
#     (src_dir / "__init__.py").write_text("", encoding="utf-8")
# if not (models_dir / "__init__.py").exists():
#     (models_dir / "__init__.py").write_text("", encoding="utf-8")

# # ------------ now imports ------------
# import argparse
# import tensorflow as tf

# # Local imports: corrected module name
# from src.data.dataloader import DataLoader
# from src.models.transfer_learing import build_transfer_model
# from src.training.callbacks import get_all_callbacks
# from src.models.save_load import save_history, ensure_dir


# def parse_args():
#     p = argparse.ArgumentParser(description="Train Transfer Learning Model for Aerial Classification")
#     p.add_argument("--data_dir", type=str, default="data/classification_dataset",
#                    help="Path to classification dataset (train/valid/test)")
#     p.add_argument("--out_dir", type=str, default="models/transfer_learning",
#                    help="Directory to save model & logs")
#     p.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
#                    help="Image size H W")
#     p.add_argument("--batch_size", type=int, default=32)
#     p.add_argument("--epochs", type=int, default=20)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--fine_tune", type=int, default=0,
#                    help="Number of top layers to unfreeze for fine-tuning")
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--use_tensorboard", action="store_true")
#     p.add_argument("--monitor", type=str, default="val_accuracy")
#     return p.parse_args()


# def main(args):
#     out_dir = ensure_dir(args.out_dir)

#     print(
#         f"\n===== Transfer Learning Training =====\n"
#         f"Data Dir      : {args.data_dir}\n"
#         f"Output Dir    : {out_dir}\n"
#         f"Image Size    : {tuple(args.img_size)}\n"
#         f"Batch Size    : {args.batch_size}\n"
#         f"Epochs        : {args.epochs}\n"
#         f"Learning Rate : {args.lr}\n"
#         f"Fine-tune     : Top {args.fine_tune} layers\n"
#     )

#     # Detect GPU
#     gpus = tf.config.list_physical_devices("GPU")
#     if gpus:
#         print(f"GPUs Detected : {[g.name for g in gpus]}")
#     else:
#         print("⚠ No GPU detected (training may be slow).")

#     # Load dataset
#     loader = DataLoader(
#         data_dir=args.data_dir,
#         img_size=tuple(args.img_size),
#         batch_size=args.batch_size,
#         seed=args.seed,
#         normalize=True,
#     )

#     train_ds, val_ds, test_ds, class_names = loader.load()
#     num_classes = len(class_names)

#     print(f"Classes found: {class_names}")

#     # Build model
#     # NOTE: build_transfer_model signature uses 'base_trainable' (not 'fine_tune_at').
#     # We set base_trainable=False (freeze base) and later unfreeze top layers manually
#     model = build_transfer_model(
#         input_shape=(args.img_size[0], args.img_size[1], 3),
#         num_classes=num_classes,
#         backbone="EfficientNetB0",
#         base_trainable=False,
#         dropout_rate=0.3,
#         pooling="avg",
#         lr=args.lr,
#         metrics=None,
#     )

#     model.summary()

#     # Apply fine-tuning (unfreeze N top layers) — manual approach
#     if args.fine_tune > 0:
#         print(f"\nUnfreezing top {args.fine_tune} layers for fine-tuning...")

#         count = 0
#         for layer in reversed(model.layers):
#             if layer.weights:
#                 layer.trainable = True
#                 count += 1
#             if count >= args.fine_tune:
#                 break

#         print(f"✓ Unfrozen layers: {count}")

#         # Lower LR for stability
#         fine_tune_lr = max(args.lr * 0.1, 1e-7)
#         print(f"Recompiling model with fine-tune LR = {fine_tune_lr}\n")

#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
#             loss=model.loss,
#             metrics=model.metrics
#         )

#     # Callbacks
#     callbacks = get_all_callbacks(
#         out_dir=str(out_dir),
#         use_tensorboard=args.use_tensorboard,
#         monitor=args.monitor
#     )

#     # Training
#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=args.epochs,
#         callbacks=callbacks
#     )

#     # Save history
#     hist_path = Path(out_dir) / "history"
#     pkl_path, json_path = save_history(history, str(hist_path))
#     print(f"\nHistory saved to:\n{pkl_path}\n{json_path}")

#     # Evaluate on test set
#     print("\nEvaluating on test set...")
#     results = model.evaluate(test_ds)
#     print("Test Results:", results)

#     # Check saved best model
#     best_model_path = Path(out_dir) / "best_model.h5"
#     if best_model_path.exists():
#         print(f"\nBest model saved at:\n{best_model_path}")
#     else:
#         fallback = Path(out_dir) / "final_model.h5"
#         print("⚠ No checkpoint found, saving final model manually...")
#         model.save(str(fallback))

#     print("\n===== Training Complete =====\n")


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
#!/usr/bin/env python3
"""
train_transfer_learning.py
--------------------------
Training script for the Transfer Learning model (EfficientNetB0 default).

This version:
 - ensures project root is on sys.path
 - auto-creates __init__.py in src/ and src/models/ if missing so imports like
   `from src.models.transfer_learning import ...` work reliably on all platforms
 - uses build_transfer_model(..., fine_tune_at=...) so fine-tuning happens inside builder
"""

# -------------------------------
# PATH FIX + ensure package __init__.py files
# -------------------------------
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

src_dir = Path(PROJECT_ROOT) / "src"
models_dir = src_dir / "models"

# Create folders if missing
src_dir.mkdir(parents=False, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# Create __init__.py files if they don't exist
if not (src_dir / "__init__.py").exists():
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
if not (models_dir / "__init__.py").exists():
    (models_dir / "__init__.py").write_text("", encoding="utf-8")

# ------------
import argparse
import tensorflow as tf

# Local imports
from src.data.dataloader import DataLoader
from src.models.transfer_learing import build_transfer_model
from src.training.callbacks import get_all_callbacks
from src.models.save_load import save_history, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train Transfer Learning Model for Aerial Classification")
    p.add_argument("--data_dir", type=str, default="data/classification_dataset",
                   help="Path to classification dataset (train/valid/test)")
    p.add_argument("--out_dir", type=str, default="models/transfer_learning",
                   help="Directory to save model & logs")
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                   help="Image size H W")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fine_tune", type=int, default=0,
                   help="Number of top layers in backbone to unfreeze for fine-tuning (0 = none)")
    p.add_argument("--backbone", type=str, default="EfficientNetB0",
                   choices=["EfficientNetB0", "MobileNetV2", "ResNet50"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--monitor", type=str, default="val_accuracy")
    return p.parse_args()


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

    # Load dataset
    loader = DataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=True,
    )

    train_ds, val_ds, test_ds, class_names = loader.load()
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # Build model (builder handles fine_tune_at)
    model = build_transfer_model(
        input_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=num_classes,
        backbone=args.backbone,
        base_trainable=False,       # freeze base initially
        fine_tune_at=(args.fine_tune if args.fine_tune > 0 else None),
        dropout_rate=0.3,
        pooling="avg",
        lr=args.lr,
        metrics=None
    )

    model.summary()

    # If we asked for fine-tuning, the builder already unfreezes top layers.
    # Recompile with a smaller LR for fine-tuning stability if fine_tune > 0
    if args.fine_tune > 0:
        ft_lr = max(args.lr * 0.1, 1e-7)
        print(f"Recompiling model for fine-tuning with lr = {ft_lr}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=ft_lr),
            loss=model.loss,
            metrics=model.metrics
        )

    # Callbacks
    callbacks = get_all_callbacks(
        out_dir=str(out_dir),
        use_tensorboard=args.use_tensorboard,
        monitor=args.monitor
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save history
    hist_path = Path(out_dir) / "history"
    pkl_path, json_path = save_history(history, str(hist_path))
    print(f"\nHistory saved to:\n{pkl_path}\n{json_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test Results:", results)

    # Check saved best model
    best_model_path = Path(out_dir) / "best_model.h5"
    if best_model_path.exists():
        print(f"\nBest model saved at:\n{best_model_path}")
    else:
        fallback = Path(out_dir) / "final_model.h5"
        print("⚠ No checkpoint found, saving final model manually...")
        model.save(str(fallback))

    print("\n===== Training Complete =====\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
