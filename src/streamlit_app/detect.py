#!/usr/bin/env python3
"""
detect.py - Streamlit page for YOLOv8 object detection using ultralytics.
"""

from pathlib import Path
import tempfile
import shutil

import streamlit as st
from PIL import Image

from ultralytics import YOLO
import utils  # local utils


DEFAULT_YOLO_WEIGHTS = "models/yolov8/run1/weights/best.pt"


def load_yolo_model(weights_path: str):
    p = Path(weights_path)
    if not p.exists():
        return None
    try:
        model = YOLO(str(weights_path))
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None


def run_yolo_on_image(
    model,
    pil_image: Image.Image,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    save_dir: str = "outputs/yolo_infer",
    save_name: str = "streamlit",
):
    """
    Runs model.predict on a single PIL image, returns results object and path to annotated image (if saved).
    """
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_img_path = tmp_dir / "input.jpg"
    pil_image.save(tmp_img_path, format="JPEG")
    try:
        results = model.predict(
            source=str(tmp_img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=True,
            project=save_dir,
            name=save_name,
            exist_ok=True,
        )
        out_dir = Path(save_dir) / save_name
        annotated = None
        for ext in [".jpg", ".png"]:
            candidates = list(out_dir.glob(f"*{ext}"))
            if candidates:
                annotated = str(candidates[0])
                break
        return results, annotated
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


def detection_page():
    st.title("Object Detection — YOLOv8")
    st.write("Upload an image for YOLOv8 detection (trained on aerial objects).")

    weights_path = st.text_input("Path to YOLO weights (.pt)", value=DEFAULT_YOLO_WEIGHTS)
    model = None
    if st.button("Load YOLO model"):
        with st.spinner("Loading YOLO model..."):
            model = load_yolo_model(weights_path)
        if model is None:
            utils.show_model_missing_message(weights_path, model_type="YOLO model")
        else:
            st.success("YOLO model loaded.")

    if "yolo_model" not in st.session_state:
        st.session_state["yolo_model"] = None
    if model is not None:
        st.session_state["yolo_model"] = model

    uploaded = st.file_uploader("Upload image for detection", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    if not utils.is_image_file(uploaded.name):
        st.error("Unsupported file type.")
        return

    pil_img = utils.read_image_from_upload(uploaded)
    utils.display_image(pil_img, caption=uploaded.name)

    model = st.session_state.get("yolo_model", None)
    if model is None:
        st.error("YOLO model not loaded. Click 'Load YOLO model' first.")
        return

    with st.spinner("Running detection..."):
        try:
            results, annotated = run_yolo_on_image(model, pil_img, imgsz=640, conf=0.25, iou=0.45)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    st.success("Inference complete.")
    if annotated:
        st.image(annotated, caption="Annotated result", use_column_width=True)
        st.write(f"Annotated image saved to: {annotated}")
    else:
        st.info("No annotated image found in output directory. Check model.predict() output for details.")

    # Quick text summary
    try:
        r = results[0]
        boxes = r.boxes
        if boxes is not None:
            n = len(boxes)
            st.write(f"Detections: {n}")
        else:
            st.write("No detections.")
    except Exception:
        pass
