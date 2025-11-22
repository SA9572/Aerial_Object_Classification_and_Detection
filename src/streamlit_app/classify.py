"""
classify.py - Streamlit page for image classification (Custom CNN / Transfer Learning).
Placed under package streamlit_app; uses absolute imports so it works when run by Streamlit.
"""

from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# <- absolute imports (NOT relative)
from streamlit_app.utils import (
    is_image_file,
    read_image_from_upload,
    show_model_missing_message,
    display_image,
)

# Default model path (you can change in the UI)
DEFAULT_MODEL_PATH = "models/transfer_learning/best_model.h5"


def load_classification_model(model_path: str):
    """
    Load a Keras model. Wrapped with st.cache_resource to avoid reloading.
    """
    if not Path(model_path).exists():
        return None
    try:
        model = tf.keras.models.load_model(str(model_path))
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
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr


def classification_page():
    st.title("Image Classification — Aerial: Bird vs Drone")
    st.write("Upload an image and the model will predict whether it's a bird or a drone.")

    model_path = st.text_input("Path to Keras model (.h5 or SavedModel folder)", value=DEFAULT_MODEL_PATH)
    model = None
    if st.button("Load model"):
        with st.spinner("Loading model..."):
            model = load_classification_model(model_path)
        if model is None:
            show_model_missing_message(model_path, model_type="Classification model")
        else:
            st.success("Model loaded successfully.")
            st.write("Model summary:")
            # Print summary to streamlit
            buf = []
            model.summary(print_fn=lambda s: buf.append(s))
            st.text("\n".join(buf))

    # Keep model in session state so user doesn't need to reload every action
    if "class_model" not in st.session_state:
        st.session_state["class_model"] = None
    if model is not None:
        st.session_state["class_model"] = model

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if not is_image_file(uploaded.name):
            st.error("Unsupported file type.")
            return
        pil_img = read_image_from_upload(uploaded)
        display_image(pil_img, caption=uploaded.name)

        st.write("Preparing image and running prediction...")
        model = st.session_state.get("class_model", None)
        if model is None:
            st.error("Model not loaded. Click 'Load model' first and wait for completion.")
            return

        img_tensor = prepare_image_for_model(pil_img, target_size=(model.input_shape[1], model.input_shape[2]))
        try:
            preds = model.predict(img_tensor)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return

        # Interpret prediction
        # handle both binary (sigmoid) and softmax outputs
        if preds.shape[-1] == 1:
            score = float(preds[0][0])
            label = "class 1" if score >= 0.5 else "class 0"
            st.write(f"Predicted: **{label}**  (score = {score:.3f})")
        else:
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            st.write(f"Predicted class index: **{class_idx}** with confidence **{confidence:.3f}**")
            st.json({"raw_output": preds.tolist()})
