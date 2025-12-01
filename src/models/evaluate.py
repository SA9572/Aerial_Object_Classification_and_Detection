"""
src/models/evaluate.py

Evaluate a trained Bird vs Drone classification model on the test set.

Assumptions:
    - Models are saved in: models/classification/*.h5
      (e.g. custom_cnn_best.h5, transfer_resnet50_best.h5, etc.)
    - Test data is organized as:

        data/
          classification_dataset/
            test/
              bird/
              drone/

Usage (from project root):

    # Evaluate a specific model
    python -m src.models.evaluate --model-path models/classification/transfer_resnet50_best.h5

    # Evaluate custom CNN
    python -m src.models.evaluate --model-path models/classification/custom_cnn_best.h5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class EvalConfig:
    model_path: Path
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    project_root: Path | None = None
    data_root: Path | None = None
    reports_dir: Path | None = None


# ---------------------------------------------------------------------
# Helper: project root & test dataset path
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def get_test_dir(project_root: Path) -> Path:
    """
    Returns the test directory inside data/classification_dataset.
    """
    test_dir = project_root / "data" / "classification_dataset" / "test"
    return test_dir


# ---------------------------------------------------------------------
# Helper: create test dataset
# ---------------------------------------------------------------------
def create_test_dataset(
    test_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
) -> tuple[tf.data.Dataset, List[str]]:
    """
    Create a tf.data Dataset for the test set from folder.

    Returns:
        test_ds: dataset of (images, labels)
        class_names: list of class names (e.g. ['bird', 'drone'])
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print("Loading test dataset from:", test_dir)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,  # important for consistent y_true vs y_pred
    )

    class_names = test_ds.class_names
    print("Class names:", class_names)
    print("num_classes:", len(class_names))

    autotune = tf.data.AUTOTUNE
    test_ds = test_ds.prefetch(autotune)

    return test_ds, class_names


# ---------------------------------------------------------------------
# Helper: load model
# ---------------------------------------------------------------------
def load_trained_model(model_path: Path) -> tf.keras.Model:
    """
    Load a trained Keras model from an .h5 file.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading model from:", model_path)
    model = load_model(model_path)
    print("✅ Model loaded.")
    return model


# ---------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------
def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    cfg: EvalConfig,
) -> None:
    """
    Evaluate the model and print metrics:
        - loss, accuracy
        - classification report
        - confusion matrix
    """
    print("\n🔍 Evaluating model on test set...\n")

    # 1) Quick evaluate via Keras
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"✅ Test Loss:     {loss:.4f}")
    print(f"✅ Test Accuracy: {acc:.4f}")

    # 2) Detailed metrics (classification_report, confusion_matrix)
    y_true = []
    y_pred = []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        preds_idx = np.argmax(preds, axis=1)

        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds_idx.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy (manual): {overall_acc:.4f}\n")

    print("📊 Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("📉 Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Optionally: save results to a text file in reports/
    if cfg.reports_dir is not None:
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = cfg.reports_dir / "evaluation_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Model path: {cfg.model_path}\n")
            f.write(f"Image size: {cfg.image_size}\n")
            f.write(f"Batch size: {cfg.batch_size}\n\n")
            f.write(f"Test Loss: {loss:.4f}\n")
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Overall Accuracy (manual): {overall_acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(
                classification_report(
                    y_true,
                    y_pred,
                    target_names=class_names,
                    zero_division=0,
                )
            )
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm))
        print("\n📝 Detailed evaluation report saved to:", report_path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Bird vs Drone classification model on the test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.h5) inside models/classification/",
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
        help="Batch size for evaluation (default: 32).",
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
    cfg = EvalConfig(
        model_path=project_root / args.model_path if not args.model_path.startswith(str(project_root)) else Path(args.model_path),
        image_size=(args.image_size[0], args.image_size[1]),
        batch_size=args.batch_size,
        project_root=project_root,
        data_root=project_root / "data",
        reports_dir=project_root / "reports",
    )

    print("📌 Evaluation configuration:")
    print(asdict(cfg))

    # Test dataset
    test_dir = get_test_dir(project_root)
    test_ds, class_names = create_test_dataset(
        test_dir=test_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )

    # Load model
    model = load_trained_model(cfg.model_path)

    # Evaluate
    evaluate_model(model, test_ds, class_names, cfg)

    print("\n✅ Evaluation finished.")


if __name__ == "__main__":
    main()
