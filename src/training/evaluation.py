"""
evaluation.py
--------------
Utilities to evaluate saved Keras classification models on the test dataset.

Provides:
 - evaluate_model_file(model_path, test_ds, class_names, out_dir)
 - get_predictions_from_dataset(model, dataset)
 - plot_confusion_matrix(cm, class_names, out_path)
 - plot_roc_pr(...)  # for binary case only

Also provides a CLI to run evaluation from the project root:
    python src/training/evaluation.py --model models/transfer_learning/best_model.h5 --data_dir data/classification_dataset --out_dir outputs/eval_tl

Notes:
 - Expects classification dataset layout: data_dir/test/<class>/*.jpg
 - Uses DataLoader for loading datasets (so normalization and img_size are consistent).
 - Requires sklearn, matplotlib, seaborn.
"""

import argparse
from pathlib import Path
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Add project root to path if needed when run as script
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import DataLoader
from src.models.save_load import load_keras_model, ensure_dir


def get_predictions_from_dataset(model, dataset):
    """
    Run model.predict over a tf.data dataset and return y_true, y_pred_labels, y_pred_probs

    Args:
        model: loaded tf.keras.Model
        dataset: tf.data.Dataset yielding (images, labels)

    Returns:
        y_true: list of int labels
        y_pred: list of predicted int labels
        y_probs: np.array of shape (N, num_classes) of predicted probabilities
    """
    y_true = []
    y_probs_list = []

    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_probs_list.append(probs)
        y_true.extend(batch_y.numpy().tolist())

    y_probs = np.vstack(y_probs_list)
    y_pred = np.argmax(y_probs, axis=1).tolist()

    return np.array(y_true), np.array(y_pred), y_probs


def save_classification_report(y_true, y_pred, class_names, out_path):
    """
    Generate sklearn classification report text and save to out_path (txt).
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def plot_confusion_matrix(cm, class_names, out_path, figsize=(6,6), cmap="Blues"):
    """
    Plot and save confusion matrix heatmap.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_pr(y_true, y_probs, class_names, out_dir):
    """
    Plot ROC curve (one-vs-rest) and PR curve for binary case.
    - If binary (num_classes==2), plots ROC and PR for positive class (class index 1).
    - For multi-class, this function will plot per-class ROC if appropriate probabilities exist.

    Saves files to out_dir: roc.png and pr.png
    """
    num_classes = y_probs.shape[1]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if num_classes == 2:
        # binary: use class 1 as positive
        y_true_bin = (y_true == 1).astype(int)
        y_score = y_probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (class=1)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "roc.png", dpi=150)
        plt.close()

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        ap = average_precision_score(y_true_bin, y_score)
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (class=1)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(out_dir / "pr.png", dpi=150)
        plt.close()
    else:
        # multiclass: plot ROC per class (requires binarized labels)
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            plt.figure(figsize=(8,6))
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC Curves (one-vs-rest)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "roc_multiclass.png", dpi=150)
            plt.close()
        except Exception:
            # skip if label binarize not available or fails
            pass


def evaluate_model_file(model_path, data_dir="data/classification_dataset", img_size=(224,224), batch_size=32, out_dir="outputs/eval"):
    """
    High-level convenience function:
      - loads Keras model from model_path
      - loads test dataset via DataLoader (with img_size and batch_size)
      - runs predictions
      - saves classification report, confusion matrix, ROC/PR (if applicable), and a CSV summary

    Returns:
        dict summary with keys: accuracy, per_class_counts, paths to saved files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = load_keras_model(str(model_path))

    # Load only test set using DataLoader to ensure same preprocessing
    loader = DataLoader(data_dir=data_dir, img_size=img_size, batch_size=batch_size, normalize=True)
    # We only need test dataset for evaluation; dataloader.load() returns train,val,test
    _, _, test_ds, class_names = loader.load()

    print("Running predictions on test set...")
    y_true, y_pred, y_probs = get_predictions_from_dataset(model, test_ds)

    # Reports & metrics
    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    acc = np.mean(y_true == y_pred)

    # Save text report
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save confusion matrix plot
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save ROC/PR if possible
    try:
        plot_roc_pr(y_true, y_probs, class_names, out_dir)
    except Exception as e:
        print("Warning: ROC/PR plotting failed:", e)

    # Save CSV summary of predictions
    import pandas as pd
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    # Add per-class probabilities as columns
    for i, name in enumerate(class_names):
        df[f"prob_{name}"] = y_probs[:, i]
    csv_path = out_dir / "predictions_summary.csv"
    df.to_csv(csv_path, index=False)

    # Save metadata
    meta = {
        "model_path": str(model_path),
        "data_dir": data_dir,
        "img_size": img_size,
        "batch_size": batch_size,
        "accuracy": float(acc),
        "num_samples": int(len(y_true)),
        "class_names": class_names,
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions_csv": str(csv_path)
    }
    meta_path = out_dir / "eval_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved report -> {report_path}")
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved predictions CSV -> {csv_path}")
    print(f"Accuracy: {acc:.4f}")

    return meta


# -----------------------
# CLI
# -----------------------
def parse_cli_args():
    p = argparse.ArgumentParser(description="Evaluate a saved Keras classification model on test dataset")
    p.add_argument("--model", required=True, help="Path to saved Keras model (.h5 or SavedModel dir)")
    p.add_argument("--data_dir", default="data/classification_dataset", help="Path to classification dataset root")
    p.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image size H W")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="outputs/eval", help="Output directory for reports & plots")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    evaluate_model_file(model_path=args.model, data_dir=args.data_dir, img_size=tuple(args.img_size),
                        batch_size=args.batch_size, out_dir=args.out_dir)
