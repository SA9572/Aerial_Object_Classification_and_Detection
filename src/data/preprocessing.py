"""
preprocessing.py
-------------------------
Reusable preprocessing utilities for:
 - classification models
 - streamlit inference
 - model training scripts

Handles loading, resizing, normalizing, converting to tensor.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


# --------------------------------------------
# ✔ Load and preprocess image for inference
# --------------------------------------------
def load_and_preprocess_image(
    image_path,
    img_size=(224, 224),
    normalize=True,
):
    """
    Loads an image from disk and preprocesses it for model prediction.

    Args:
        image_path (str or Path): Path to the image
        img_size (tuple): (height, width)
        normalize (bool): Normalize pixel values to [0,1]

    Returns:
        preprocessed image as a 4D tensor (1, H, W, 3)
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_size)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize to 0–1
    if normalize:
        img_array = img_array / 255.0

    # Expand dims → (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --------------------------------------------
# ✔ Preprocess a PIL Image directly (Streamlit)
# --------------------------------------------
def preprocess_pil_image(
    pil_image,
    img_size=(224, 224),
    normalize=True,
):
    """
    Preprocess a PIL.Image (useful for Streamlit file uploader).

    Args:
        pil_image (PIL.Image): uploaded image
        img_size (tuple): Resize target
        normalize (bool): scale to [0,1]

    Returns:
        4D tensor (1, H, W, 3)
    """

    img = pil_image.convert("RGB")
    img = img.resize(img_size)

    img_array = np.array(img, dtype=np.float32)

    if normalize:
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --------------------------------------------
# ✔ Preprocessing layer for tf.data pipelines
# --------------------------------------------
def get_preprocessing_layer(normalize=True):
    """
    Returns a preprocessing layer to use inside Keras models.

    Args:
        normalize (bool): Apply 1./255 normalization

    Returns:
        keras.Sequential preprocessing model
    """

    layers = []

    if normalize:
        layers.append(tf.keras.layers.Rescaling(1.0 / 255.0))

    preprocessing = tf.keras.Sequential(layers, name="preprocessing")

    return preprocessing


# --------------------------------------------
# ✔ Convert prediction output into readable form
# --------------------------------------------
def decode_prediction(prediction_tensor):
    """
    Converts model output to class and confidence.

    Args:
        prediction_tensor (np.ndarray or tensor):
            Shape: (1, num_classes)

    Returns:
        (class_index, confidence)
    """

    preds = np.array(prediction_tensor)[0]  # shape: (num_classes,)
    class_idx = preds.argmax()
    confidence = preds[class_idx]

    return class_idx, confidence
