# demo_run.py
import numpy as np
from src.models.custom_cnn import build_custom_cnn

model = build_custom_cnn(input_shape=(224,224,3), num_classes=2, lr=1e-3)
model.summary()

# Dummy forward pass
dummy = np.random.rand(2,224,224,3).astype("float32")
preds = model.predict(dummy)
print("Output shape:", preds.shape)
print(preds[:2])
