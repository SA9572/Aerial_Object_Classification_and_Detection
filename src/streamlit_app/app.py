#!/usr/bin/env python3
"""
Main Streamlit app - multipage style (Classification | Detection | About)

Run from project root:
    streamlit run src/streamlit_app/app.py
"""

import streamlit as st

import classify   # local module in same folder
import detect     # local module in same folder
import utils      # local module in same folder


st.set_page_config(page_title="Aerial Object - Demo", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Classification", "Detection", "About"])

if page == "Home":
    st.title("Aerial Object Classification & Detection")
    st.write(
        """
        This demo app includes:
          - Image classification (Custom CNN / Transfer Learning)
          - Object detection (YOLOv8)
        Use the sidebar to switch pages.
        """
    )
    st.write("Quick links:")
    st.markdown("- **Classification**: upload an image and a Keras model to classify.")
    st.markdown("- **Detection**: upload an image and a YOLOv8 `.pt` model to detect objects (bird/drone).")

elif page == "Classification":
    classify.classification_page()

elif page == "Detection":
    detect.detection_page()

elif page == "About":
    st.title("About this project")
    st.write("Project PDF (uploaded) and basic instructions.")
    utils.add_pdf_download_button()
    st.markdown("### Notes")
    st.write("- Make sure TensorFlow and `ultralytics` are installed in your environment.")
    st.write("- Models should be saved under `models/transfer_learning` and `models/yolov8` respectively by default.")
    st.write("- Run the app from the *project root* so relative paths resolve correctly.")
