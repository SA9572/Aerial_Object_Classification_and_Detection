"""
save_load.py
------------
Utility helpers to save and load Keras models and training histories.

Provides:
 - save_keras_model(model, out_dir, name="best_model", save_format="h5")
 - load_keras_model(model_path)
 - save_history(history, out_path)
 - load_history(in_path)
 - ensure_dir(path)

Notes:
 - save_format: "h5" will save as HDF5 (.h5). "tf" will save TensorFlow SavedModel format (folder).
 - Use `save_keras_model(..., save_format="tf")` if you want SavedModel for TF serving.
"""

import os
from pathlib import Path
import json
import pickle
import tensorflow as tf


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_keras_model(model: tf.keras.Model, out_dir: str, name: str = "best_model", save_format: str = "h5"):
    """
    Save a Keras model either as HDF5 (.h5) or as TensorFlow SavedModel folder.

    Args:
        model: tf.keras.Model instance
        out_dir: output directory where model will be saved
        name: base name for the saved file/folder
        save_format: "h5" or "tf"

    Returns:
        Path to saved model
    """
    out_dir = ensure_dir(out_dir)
    if save_format == "h5":
        out_path = out_dir / f"{name}.h5"
        model.save(str(out_path), save_format="h5")
    elif save_format == "tf":
        out_path = out_dir / f"{name}_savedmodel"
        model.save(str(out_path), save_format="tf")
    else:
        raise ValueError("save_format must be 'h5' or 'tf'")

    return str(out_path)


def load_keras_model(model_path: str):
    """
    Load a Keras model from .h5 or SavedModel directory.

    Args:
        model_path: path to .h5 file or SavedModel folder

    Returns:
        Loaded tf.keras.Model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    # Let tf.keras handle both .h5 and SavedModel
    model = tf.keras.models.load_model(str(model_path))
    return model


def save_history(history, out_path: str):
    """
    Save training history (history.history dict) to pickle and json.

    Args:
        history: Keras History object OR dict-like (history.history)
        out_path: file path (without extension recommended) or directory

    Returns:
        Tuple (pickle_path, json_path)
    """
    out_path = Path(out_path)
    # If out_path is a directory, use default base name
    if out_path.is_dir():
        out_path = out_path / "history"

    # Extract dict
    hist_dict = history.history if hasattr(history, "history") else dict(history)

    pkl_path = out_path.with_suffix(".pkl")
    json_path = out_path.with_suffix(".json")

    with open(pkl_path, "wb") as f:
        pickle.dump(hist_dict, f)

    # convert numpy arrays to lists for json serialization
    serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in hist_dict.items()}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return str(pkl_path), str(json_path)


def load_history(in_path: str):
    """
    Load a saved history. Accepts .pkl (preferred) or .json.

    Args:
        in_path: path to .pkl OR .json file

    Returns:
        dict of history values
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"History file not found: {in_path}")

    if in_path.suffix == ".pkl":
        with open(in_path, "rb") as f:
            hist = pickle.load(f)
    elif in_path.suffix == ".json":
        import json
        with open(in_path, "r") as f:
            hist = json.load(f)
    else:
        raise ValueError("Unsupported file format for history. Use .pkl or .json")

    return hist


# -----------------------
# Example usage (commented)
# -----------------------
# from src.models.save_load import save_keras_model, load_keras_model, save_history, load_history
# model = build_custom_cnn(...)  # your model
# save_keras_model(model, "models/custom_cnn", name="best_model", save_format="h5")
#
# # After training:
# save_history(history, "models/custom_cnn/history")
#
# # Load:
# model = load_keras_model("models/custom_cnn/best_model.h5")
# hist = load_history("models/custom_cnn/history.pkl")
