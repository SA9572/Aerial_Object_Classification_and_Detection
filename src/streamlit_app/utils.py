"""
utils.py - shared Streamlit helpers for model loading, preprocessing, and safe file handling.
"""

from pathlib import Path
from PIL import Image
import streamlit as st

# Path to uploaded project PDF (if present in this environment)
UPLOADED_PDF = "/mnt/data/Project Title.pdf"

# ---------------------------
# File helpers
# ---------------------------
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXT


def read_image_from_upload(uploaded_file) -> Image.Image:
    """
    uploaded_file: streamlit UploadedFile
    returns: PIL.Image (RGB)
    """
    image = Image.open(uploaded_file).convert("RGB")
    return image


# ---------------------------
# Streamlit UI helpers
# ---------------------------
def show_model_missing_message(model_path: str, model_type: str = "model"):
    st.error(f"⚠️ {model_type} not found at: {model_path}")
    st.info("Make sure you trained the model and saved it at this path, or update the app settings.")
    st.caption("Fix the path and rerun the app.")


def add_pdf_download_button():
    """
    Adds a download button for the uploaded project PDF (if present).
    """
    p = Path(UPLOADED_PDF)
    if p.exists():
        with open(p, "rb") as f:
            data = f.read()
        st.download_button(
            label="Download project PDF",
            data=data,
            file_name=p.name,
            mime="application/pdf",
        )
    else:
        st.info("Project PDF not found on the server.")


# ---------------------------
# Image display helper
# ---------------------------
def display_image(img: Image.Image, caption: str | None = None, width: int | None = None):
    st.image(img, caption=caption, use_container_width=(width is None))
