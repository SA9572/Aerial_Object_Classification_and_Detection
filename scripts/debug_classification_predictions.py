#!/usr/bin/env python3
"""
Debug script: load best_model.h5 and print predictions for several test images.
Run from project root:

    python scripts/debug_classification_predictions.py
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# ===== CONFIGURE THESE =====
MODEL_PATH = "models/transfer_learning/best_model.h5"
TEST_DIR = "data/classification_dataset/test"  # expects test/<class>/*.jpg
IMG_SIZE = (224, 224)
CLASS_NAMES = ["bird", "drone"]  # adjust to match your class order
# ===========================


def load_image(path, target_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0  # match your DataLoader+model scale
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print("Model not found:", model_path)
        return

    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print("Model loaded.")

    test_root = Path(TEST_DIR)
    if not test_root.exists():
        print("Test directory not found:", test_root)
        return

    # collect some test images
    img_paths = list(test_root.rglob("*.jpg")) + list(test_root.rglob("*.png")) + list(test_root.rglob("*.jpeg"))
    img_paths = img_paths[:20]  # only first 20 for debug
    if not img_paths:
        print("No test images found under:", test_root)
        return

    print(f"\nTesting on {len(img_paths)} images:\n")
    for p in img_paths:
        x = load_image(p, IMG_SIZE)
        preds = model.predict(x, verbose=0)
        if preds.shape[-1] == 1:
            score = float(preds[0][0])
            print(f"{p}: score={score:.4f}")
        else:
            probs = preds[0]
            class_idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            if class_idx < len(CLASS_NAMES):
                label = CLASS_NAMES[class_idx]
            else:
                label = f"class_{class_idx}"
            print(f"{p}: {label} (idx={class_idx}, conf={conf:.3f}), probs={probs}")
    print("\nDone.")


if __name__ == "__main__":
    main()
