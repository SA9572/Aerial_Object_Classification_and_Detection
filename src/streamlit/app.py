"""
Streamlit app for Aerial Object Classification & Detection
----------------------------------------------------------

Features:
    - Image Classification (Bird vs Drone) using best Keras model:
        models/classification/transfer_resnet50_best.h5

    - Optional Object Detection (YOLOv8) using:
        models/detection/yolov8n_best.pt
      (if ultralytics is installed and model file exists)

Run from project root:

    streamlit run src/streamlit/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------------
# Fix Python path so `src` package is importable under Streamlit
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
# project root assumed to be two levels up: <root>/src/streamlit/app.py
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now imports from `src` will work
from src.deployment.predict import (  # type: ignore
    load_model_and_classes,
    predict_image,
)

# YOLO (optional)
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    YOLO_AVAILABLE = False


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def get_project_root() -> Path:
    """
    Return the project root that we already computed.
    """
    return PROJECT_ROOT


CLASS_MODEL_DEFAULT = PROJECT_ROOT / "models" / "classification" / "transfer_resnet50_best.h5"
YOLO_MODEL_DEFAULT = PROJECT_ROOT / "models" / "detection" / "yolov8n_best.pt"


# ---------------------------------------------------------------------
# Streamlit caching: load models once
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_classification_model():
    """
    Load Keras classification model + class names once per session.
    Uses:
        model_type="transfer", backbone="resnet50"
        models/classification/transfer_resnet50_best.h5
    """
    st.write("🔁 Loading classification model (ResNet50 transfer)...")
    model, class_names, cfg = load_model_and_classes(
        model_type="transfer",
        backbone="resnet50",
        model_path=str(CLASS_MODEL_DEFAULT) if CLASS_MODEL_DEFAULT.exists() else None,
        image_size=(224, 224),
        project_root=PROJECT_ROOT,
    )
    return model, class_names, cfg


@st.cache_resource(show_spinner=True)
def get_yolo_model():
    """
    Load YOLOv8 detection model once per session.
    Uses:
        models/detection/yolov8n_best.pt
    """
    if not YOLO_AVAILABLE:
        return None, "❌ ultralytics is not installed. Install with: pip install ultralytics"

    if not YOLO_MODEL_DEFAULT.exists():
        return None, f"❌ YOLO model not found at: {YOLO_MODEL_DEFAULT}"

    try:
        model = YOLO(str(YOLO_MODEL_DEFAULT))
        return model, None
    except Exception as e:
        return None, f"❌ Failed to load YOLO model: {e}"


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def show_sidebar_info():
    st.sidebar.title("⚙️ Settings")

    st.sidebar.markdown(
        """
        **Project:** Aerial Object Classification & Detection  
        - 🐦 Bird vs 🚁 Drone  
        - 🎯 Classification (ResNet50 TL)  
        - 📦 Optional YOLOv8 Detection
        """
    )

    mode = st.sidebar.radio(
        "Choose Mode:",
        ["Classification", "Detection (YOLOv8)"],
        index=0,
    )

    conf_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Used only in YOLO Detection mode",
    )

    return mode, conf_threshold


def pil_image_from_uploaded(file) -> Image.Image:
    """
    Convert uploaded file to a PIL image.
    """
    bytes_data = file.read()
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    return img


def display_prediction_probs(class_names: List[str], probs: np.ndarray):
    st.subheader("📊 Class Probabilities")
    for i, p in enumerate(probs):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        st.write(f"- **{name}**: `{p:.4f}`")


# ---------------------------------------------------------------------
# Classification workflow
# ---------------------------------------------------------------------
def run_classification_mode():
    st.header("🐦🚁 Bird vs Drone Classification")

    st.markdown(
        """
        Upload an aerial image, and the model will classify it as
        **Bird** or **Drone** using a transfer-learning ResNet50 classifier.
        """
    )

    uploaded = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if not uploaded:
        st.info("👆 Please upload an image to get a prediction.")
        return

    # Show image
    img = pil_image_from_uploaded(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load model
    try:
        model, class_names, cfg = get_classification_model()
    except Exception as e:
        st.error(f"❌ Failed to load classification model: {e}")
        return

    if st.button("🔮 Run Classification"):
        with st.spinner("Running classification..."):
            tmp_dir = PROJECT_ROOT / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / "tmp_uploaded_image.jpg"
            img.save(tmp_path)

            pred_label, pred_idx, probs = predict_image(
                model,
                tmp_path,
                class_names=class_names,
                target_size=cfg.image_size,
            )

        st.success(f"✅ Predicted: **{pred_label}** (class index: {pred_idx})")
        display_prediction_probs(class_names, probs)


# ---------------------------------------------------------------------
# YOLO Detection workflow
# ---------------------------------------------------------------------
def run_detection_mode(conf_threshold: float):
    st.header("📦 YOLOv8 Object Detection (Bird / Drone)")

    st.markdown(
        """
        Upload an image to run **YOLOv8** detection.  
        Bounding boxes and labels for **Bird** / **Drone** will be drawn.
        """
    )

    uploaded = st.file_uploader(
        "Upload an image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="yolo_uploader",
    )

    if not uploaded:
        st.info("👆 Please upload an image to run YOLO detection.")
        return

    # Show image
    img = pil_image_from_uploaded(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load YOLO model
    model, err = get_yolo_model()
    if model is None:
        st.error(err)
        return

    if st.button("🧭 Run YOLO Detection"):
        with st.spinner("Running YOLOv8 detection..."):
            tmp_dir = PROJECT_ROOT / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / "tmp_yolo_image.jpg"
            img.save(tmp_path)

            results = model.predict(
                source=str(tmp_path),
                imgsz=640,
                conf=conf_threshold,
                device="cpu",  # change to "0" if you want GPU
                save=False,
                verbose=False,
            )

        if not results:
            st.warning("No detections returned.")
            return

        res = results[0]
        plotted = res.plot()  # RGB numpy array with boxes
        st.image(plotted, caption="YOLOv8 Detection Result", use_column_width=True)

        if res.boxes is not None and res.boxes.cls is not None:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
            st.subheader("📊 Detections:")
            for cid, score in zip(cls_ids, scores):
                st.write(f"- Class ID {cid} → conf: `{float(score):.3f}`")
        else:
            st.info("No objects detected above the confidence threshold.")


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Aerial Object Classification & Detection",
        page_icon="🛰️",
        layout="centered",
    )

    st.title("🛰️ Aerial Object Classification & Detection")
    st.caption("Bird vs Drone • ResNet50 Classification • YOLOv8 Detection (Optional)")

    mode, conf_threshold = show_sidebar_info()

    if mode == "Classification":
        run_classification_mode()
    else:
        run_detection_mode(conf_threshold)


if __name__ == "__main__":
    main()
