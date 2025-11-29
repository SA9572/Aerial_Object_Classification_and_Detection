#!/usr/bin/env python3
"""
classify.py - Streamlit page for image classification (Custom CNN / Transfer Learning).

This file is meant to be imported by app.py:
    import classify
    classify.classification_page()
"""

from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

import utils  # local module in same folder


# Default model path (you can change this in the UI)
DEFAULT_MODEL_PATH = "models/transfer_learning/best_model.h5"
CLASS_NAMES = ["bird", "drone"]  # index 0 -> bird, 1 -> drone



def load_classification_model(model_path: str):
    """
    Load a Keras model from disk for inference only (compile=False).
    This avoids issues restoring optimizer/metrics when we only need predict().
    """
    if not Path(model_path).exists():
        return None
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def prepare_image_for_model(pil_image: Image.Image, target_size=(224, 224)):
    """
    Convert PIL image to float32 batch tensor normalized to [0,1]
    """
    img = pil_image.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr


def classification_page():
    """
    Main Streamlit page for classification. Call this from app.py.
    """
    st.title("Image Classification — Aerial: Bird vs Drone")
    st.write("Upload an image and the model will predict whether it's a bird or a drone.")

    model_path = st.text_input(
        "Path to Keras model (.h5 or SavedModel folder)",
        value=DEFAULT_MODEL_PATH,
    )
    model = None

    if st.button("Load model"):
        with st.spinner("Loading model..."):
            model = load_classification_model(model_path)

        if model is None:
            utils.show_model_missing_message(model_path, model_type="Classification model")
        else:
            st.success("Model loaded successfully.")
            st.write("Model summary:")
            buf = []
            model.summary(print_fn=lambda s: buf.append(s))
            st.text("\n".join(buf))

    # Keep model in session state so we don't reload on every interaction
    if "class_model" not in st.session_state:
        st.session_state["class_model"] = None
    if model is not None:
        st.session_state["class_model"] = model

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    if not utils.is_image_file(uploaded.name):
        st.error("Unsupported file type.")
        return

    pil_img = utils.read_image_from_upload(uploaded)
    utils.display_image(pil_img, caption=uploaded.name)

    st.write("Preparing image and running prediction...")
    model = st.session_state.get("class_model", None)
    if model is None:
        st.error("Model not loaded. Click 'Load model' first and wait for completion.")
        return

    # Use the model's input size if available, otherwise default to 224x224
    try:
        target_h, target_w = model.input_shape[1], model.input_shape[2]
    except Exception:
        target_h, target_w = 224, 224

    img_tensor = prepare_image_for_model(pil_img, target_size=(target_h, target_w))

    try:
        preds = model.predict(img_tensor)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        return

    # Interpret prediction
    if preds.shape[-1] == 1:  # binary sigmoid
        score = float(preds[0][0])
        label = "class 1" if score >= 0.5 else "class 0"
        st.write(f"Predicted: **{label}**  (score = {score:.3f})")
    else:  # softmax multiclass
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        label_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"
        st.write(f"Predicted: **{label_name}**  (confidence = {confidence:.3f})")
        # Optional: show raw probabilities per class
        probs = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        st.write("Class probabilities:")
        st.json(probs)



if __name__ == "__main__":
    print("This module is intended to be used via Streamlit:")
    print("  streamlit run src/streamlit_app/app.py")
