# tests/test_preprocessing.py
from src.data.preprocessing import load_and_preprocess_image, preprocess_pil_image, decode_prediction
from PIL import Image
import numpy as np

# replace with a real image path present in your dataset
img_path = "data/classification_dataset/test/bird/0001.jpg"

arr = load_and_preprocess_image(img_path, img_size=(224,224))
print("Loaded shape:", arr.shape, "dtype:", arr.dtype, "min/max:", arr.min(), arr.max())

# PIL test:
img = Image.open(img_path)
arr2 = preprocess_pil_image(img, img_size=(224,224))
print("PIL preprocess shape:", arr2.shape)
