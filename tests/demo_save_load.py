"""
demo_save_load_fixed.py

Fixed demo to avoid metrics shape mismatch error during training.

- Builds the custom CNN
- Recompiles it with a safe metric (accuracy) to avoid Keras confusion-matrix shape issues
- Trains 1 epoch on tiny synthetic data
- Saves & loads model + history
- Runs a dummy inference

Run:
    python demo_save_load_fixed.py
"""

import os
import numpy as np

# Try to import TF-related modules; give a friendly error if TF is missing.
try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow is required to run this demo. Install it first (e.g. `pip install tensorflow`) "
        f"Original import error: {e}"
    )

from src.models.custom_cnn import build_custom_cnn
from src.models.save_load import save_keras_model, load_keras_model, save_history, load_history, ensure_dir

def make_synthetic_dataset(num_samples=64, img_size=(224,224,3), num_classes=2, batch_size=8):
    """
    Create a tiny synthetic dataset for a quick demo/training.
    Returns tf.data.Dataset objects for train & val.
    """
    # create float images in [0,1]
    X = np.random.rand(num_samples, *img_size).astype("float32")
    # integer labels 0..num_classes-1, shape (num_samples,)
    y = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)

    # split
    split = int(num_samples * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # ensure shapes are consistent
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(32).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def main():
    project_root = os.getcwd()
    models_dir = os.path.join(project_root, "models", "custom_cnn")
    ensure_dir(models_dir)

    print("Building model...")
    # build model (this function compiles it initially with precision/recall in original file)
    model = build_custom_cnn(input_shape=(224,224,3), num_classes=2, lr=1e-3)

    # --- FIX: recompile with safe metrics to avoid the Keras confusion/shape bug for this demo ---
    print("Recompiling model with safe metric ['accuracy'] for demo training...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    # -------------------------------------------------------------------------------

    model.summary()

    # Prepare tiny synthetic data for 1-epoch run (fast)
    print("Preparing synthetic dataset...")
    train_ds, val_ds = make_synthetic_dataset(num_samples=64, img_size=(224,224,3), num_classes=2, batch_size=8)

    print("Training for 1 epoch (quick demo)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)

    # Save model & history
    print("Saving model (HDF5) and history...")
    saved_model_path = save_keras_model(model, models_dir, name="demo_best_model", save_format="h5")
    pkl_path, json_path = save_history(history, os.path.join(models_dir, "demo_history"))

    print(f"Model saved to: {saved_model_path}")
    print(f"History saved to: {pkl_path} and {json_path}")

    # Load model & history back
    print("Loading model back...")
    loaded_model = load_keras_model(saved_model_path)
    print("Loaded model summary (short):")
    loaded_model.summary()

    print("Loading history back...")
    loaded_hist = load_history(pkl_path)
    print("History keys:", list(loaded_hist.keys()))

    # Run dummy inference
    print("Running a dummy forward pass...")
    dummy = np.random.rand(2,224,224,3).astype("float32")
    preds = loaded_model.predict(dummy, verbose=0)
    print("Predictions shape:", preds.shape)
    print("Predictions (first 2):")
    print(preds[:2])

    print("Demo finished successfully.")

if __name__ == "__main__":
    main()
