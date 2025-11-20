# #!/usr/bin/env python3
# """
# train_custom_cnn.py (patched)
# -----------------------------
# Train script for the custom CNN model with robust imports and debug helpers.

# Fixes included:
#  - Ensures project root is on sys.path so `from src...` imports work when running directly.
#  - Adds label-shaping map to ensure y is int32 and 1-D.
#  - Adds --debug flag that prints shapes for a few batches and enables safer metric config.
#  - Catches metric/shape errors during training and prints detailed shape diagnostics.

# Usage (recommended from project root):
#     python -m src.training.train_custom_cnn --data_dir data/classification_dataset --out_dir models/custom_cnn --epochs 25 --batch_size 32

# Or direct (less preferred) if you prefer:
#     python src/training/train_custom_cnn.py ...

# Notes:
#  - Requires TensorFlow installed in your environment.
# """

# import os
# import sys
# from pathlib import Path
# import argparse
# import traceback

# # ---------------------------
# # Put project root on sys.path
# # ---------------------------
# # This makes `from src...` imports work whether script is run directly or via -m.
# project_root = Path(__file__).resolve().parents[2]  # two levels up from src/training/
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# # Now local imports
# import tensorflow as tf

# from src.data.dataloader import DataLoader
# from src.models.custom_cnn import build_custom_cnn
# from src.training.callbacks import get_all_callbacks
# from src.models.save_load import save_history, ensure_dir

# # ---------------------------
# # CLI
# # ---------------------------
# def parse_args():
#     p = argparse.ArgumentParser(description="Train Custom CNN for Aerial Classification (patched)")
#     p.add_argument("--data_dir", type=str, default="data/classification_dataset", help="Path to classification dataset (train/valid/test)")
#     p.add_argument("--out_dir", type=str, default="models/custom_cnn", help="Output directory for model & logs")
#     p.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image size H W")
#     p.add_argument("--batch_size", type=int, default=32)
#     p.add_argument("--epochs", type=int, default=30)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--use_tensorboard", action="store_true")
#     p.add_argument("--monitor", type=str, default="val_accuracy")
#     p.add_argument("--debug", action="store_true", help="Enable debug printing and safer metric config")
#     p.add_argument("--no_cache", action="store_true", help="Disable dataset.cache() to avoid cache-related warnings while debugging")
#     return p.parse_args()

# # ---------------------------
# # Utilities to fix labels & debug
# # ---------------------------
# def make_label_safe_map():
#     import tensorflow as tf
#     def _map(x, y):
#         # Ensure integer dtype
#         y = tf.cast(y, tf.int32)
#         # If shape is (batch,1) -> squeeze to (batch,)
#         if y.shape.rank == 2 and y.shape[-1] == 1:
#             y = tf.squeeze(y, axis=-1)
#         # If one-hot vectors -> convert to indices
#         if y.shape.rank == 2 and y.shape[-1] > 1:
#             y = tf.argmax(y, axis=-1)
#             y = tf.cast(y, tf.int32)
#         return x, y
#     return _map

# def debug_print_batch_shapes(train_ds, val_ds, test_ds, model):
#     print("\n--- DEBUG: Inspecting dataset & model shapes (one batch each) ---")
#     for name, ds in [("train", train_ds), ("valid", val_ds), ("test", test_ds)]:
#         try:
#             batch = next(iter(ds))
#             x_batch, y_batch = batch
#             print(f"{name} x.shape={getattr(x_batch,'shape',None)}, x.dtype={getattr(x_batch,'dtype',None)}")
#             print(f"{name} y.shape={getattr(y_batch,'shape',None)}, y.dtype={getattr(y_batch,'dtype',None)}")
#             # model forward pass
#             preds = model(x_batch, training=False)
#             print(f"{name} model_pred.shape={getattr(preds,'shape',None)}, pred.dtype={getattr(preds,'dtype',None)}")
#         except Exception as e:
#             print(f"Failed to inspect {name} dataset: {e}")
#             traceback.print_exc()
#         print("-"*60)

# # ---------------------------
# # Main
# # ---------------------------
# def main(args):
#     out_dir = ensure_dir(args.out_dir)
#     print(f"Training custom CNN\nData dir: {args.data_dir}\nOutput dir: {out_dir}\nImage size: {tuple(args.img_size)}\nBatch size: {args.batch_size}\nEpochs: {args.epochs}\nLR: {args.lr}\nDebug: {args.debug}\nNo cache: {args.no_cache}")

#     # GPU info
#     gpus = tf.config.list_physical_devices("GPU")
#     if gpus:
#         print("GPUs detected:", [g.name for g in gpus])
#     else:
#         print("No GPU detected. Training will run on CPU unless configured.")

#     # Load datasets
#     loader = DataLoader(
#         data_dir=args.data_dir,
#         img_size=tuple(args.img_size),
#         batch_size=args.batch_size,
#         seed=args.seed,
#         normalize=True
#     )

#     # If user wants to disable cache for debugging, temporarily modify DataLoader behavior by re-loading and disabling cache manually.
#     train_ds, val_ds, test_ds, class_names = loader.load()

#     # Optionally disable cache (helps avoid "partially cached" warnings while debugging)
#     if args.no_cache:
#         # remove cache by recreating dataset from directory without caching layer
#         from tensorflow.keras.preprocessing import image_dataset_from_directory
#         train_ds = image_dataset_from_directory(args.data_dir + "/train", labels="inferred", label_mode="int",
#                                                 image_size=tuple(args.img_size), batch_size=args.batch_size, shuffle=True, seed=args.seed)
#         val_ds = image_dataset_from_directory(args.data_dir + "/valid", labels="inferred", label_mode="int",
#                                               image_size=tuple(args.img_size), batch_size=args.batch_size, shuffle=False, seed=args.seed)
#         test_ds = image_dataset_from_directory(args.data_dir + "/test", labels="inferred", label_mode="int",
#                                                image_size=tuple(args.img_size), batch_size=args.batch_size, shuffle=False, seed=args.seed)

#         # apply simple normalization
#         normalization = tf.keras.layers.Rescaling(1.0/255.0)
#         train_ds = train_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
#         val_ds = val_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
#         test_ds = test_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

#     # Ensure labels are safe (1-D int)
#     safe_map = make_label_safe_map()
#     train_ds = train_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)
#     val_ds = val_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)
#     test_ds = test_ds.map(safe_map, num_parallel_calls=tf.data.AUTOTUNE)

#     # Prefetch already applied by DataLoader; but ensure datasets are optimized
#     train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
#     val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
#     test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

#     print(f"Classes: {class_names} (num_classes={len(class_names)})")

#     # Build model
#     num_classes = len(class_names)
#     model = build_custom_cnn(input_shape=(args.img_size[0], args.img_size[1], 3), num_classes=num_classes, lr=args.lr)
#     model.summary()

#     # If debug, print one batch shapes and use safer metric list (start with accuracy)
#     if args.debug:
#         debug_print_batch_shapes(train_ds, val_ds, test_ds, model)
#         # compile with only accuracy to quickly detect if metrics cause shape mismatch
#         print("WARNING: Debug mode - compiling model with metrics=['accuracy'] to isolate metric issues.")
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
#                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     else:
#         # normal compile (build_custom_cnn already compiled, but recompile to ensure lr and metrics if desired)
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
#                       loss="sparse_categorical_crossentropy",
#                       metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

#     # Callbacks
#     callbacks = get_all_callbacks(out_dir=str(out_dir), use_tensorboard=args.use_tensorboard, monitor=args.monitor)

#     # Training with exception handling to print shape diagnostics if TF raises a shape error
#     try:
#         history = model.fit(
#             train_ds,
#             validation_data=val_ds,
#             epochs=args.epochs,
#             callbacks=callbacks
#         )
#     except tf.errors.InvalidArgumentError as e:
#         # TensorFlow shape error - print diagnostics
#         print("\n--- TensorFlow InvalidArgumentError during training ---")
#         print("Error message:")
#         print(e)
#         print("\nCollecting diagnostic shapes from a few batches to help debug...\n")
#         try:
#             for i, (x_batch, y_batch) in enumerate(train_ds.take(4)):
#                 preds = None
#                 try:
#                     preds = model(x_batch, training=False)
#                 except Exception as e_inner:
#                     print(f"[batch {i}] model forward pass failed: {e_inner}")
#                 print(f"[batch {i}] x.shape={getattr(x_batch,'shape',None)}, x.dtype={getattr(x_batch,'dtype',None)}")
#                 print(f"[batch {i}] y.shape={getattr(y_batch,'shape',None)}, y.dtype={getattr(y_batch,'dtype',None)}")
#                 print(f"[batch {i}] preds.shape={getattr(preds,'shape',None) if preds is not None else 'forward-failed'}")
#                 print("-"*40)
#         except Exception as diag_e:
#             print("Failed to collect further diagnostics:", diag_e)
#         # Re-raise to allow normal stack trace after diagnostics
#         raise

#     # Save history & evaluate on test set
#     pkl_path, json_path = save_history(history, str(Path(out_dir) / "history"))
#     print(f"Saved history: {pkl_path}, {json_path}")

#     print("\nEvaluating on test set...")
#     results = model.evaluate(test_ds)
#     print("Test results (loss & metrics):", results)

#     best_model_path = Path(out_dir) / "best_model.h5"
#     if best_model_path.exists():
#         print(f"Best model saved at: {best_model_path}")
#     else:
#         fallback = str(Path(out_dir) / "final_model.h5")
#         print("No checkpoint found, saving final model to", fallback)
#         model.save(fallback)

#     print("Training complete.")

# # ---------------------------
# # Entry point
# # ---------------------------
# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
#!/usr/bin/env python3
"""
train_custom_cnn.py
--------------------
Training script for the Custom CNN model.

Features:
 - PATH fix so `src` imports reliably
 - CLI arguments for dataset/outdir/img_size/epochs/lr etc.
 - Uses DataLoader, centralized callbacks, and save_history
 - Compiles model with SparseCategoricalAccuracy (safe for multiclass)
"""

# PATH fix and package init
import sys, os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

src_dir = Path(PROJECT_ROOT) / "src"
models_dir = src_dir / "models"
src_dir.mkdir(parents=False, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
if not (src_dir / "__init__.py").exists():
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
if not (models_dir / "__init__.py").exists():
    (models_dir / "__init__.py").write_text("", encoding="utf-8")

# Imports
import argparse
import tensorflow as tf

from src.data.dataloader import DataLoader
from src.models.custom_cnn import build_custom_cnn
from src.training.callbacks import get_all_callbacks
from src.models.save_load import save_history, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train Custom CNN for Aerial Classification")
    p.add_argument("--data_dir", type=str, default="data/classification_dataset",
                   help="Path to classification dataset (train/valid/test)")
    p.add_argument("--out_dir", type=str, default="models/custom_cnn",
                   help="Directory to save model & logs")
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224], help="Image H W")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_tensorboard", action="store_true")
    p.add_argument("--monitor", type=str, default="val_accuracy")
    return p.parse_args()


def main(args):
    out_dir = ensure_dir(args.out_dir)

    print(
        f"\n===== Custom CNN Training =====\n"
        f"Data Dir : {args.data_dir}\n"
        f"Out Dir  : {out_dir}\n"
        f"Img Size : {tuple(args.img_size)}\n"
        f"Batch    : {args.batch_size}\n"
        f"Epochs   : {args.epochs}\n"
        f"LR       : {args.lr}\n"
    )

    # GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs: {[g.name for g in gpus]}")
    else:
        print("⚠ No GPU detected (training on CPU)")

    # Data
    loader = DataLoader(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        normalize=True,
    )
    train_ds, val_ds, test_ds, class_names = loader.load()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # Build model
    model = build_custom_cnn(
        input_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=num_classes,
        lr=args.lr,
    )

    model.summary()

    # Callbacks
    callbacks = get_all_callbacks(out_dir=str(out_dir), use_tensorboard=args.use_tensorboard, monitor=args.monitor)

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
    print(f"History saved: {pkl_path}, {json_path}")

    # Evaluate
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test results:", results)

    # Check model checkpoint
    best_model_path = Path(out_dir) / "best_model.h5"
    if best_model_path.exists():
        print(f"Best model at: {best_model_path}")
    else:
        fallback = Path(out_dir) / "final_model.h5"
        print("No checkpoint found, saving final model...")
        model.save(str(fallback))

    print("\n===== Training Complete =====\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
