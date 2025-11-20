"""
callbacks.py
-----------------------
Centralized callback utilities used for training CNN and Transfer Learning models.

Provides:
 - get_basic_callbacks(...)
 - get_tensorboard_callback(...)
 - get_all_callbacks(...)
"""

from pathlib import Path
import tensorflow as tf


def ensure_dir(path):
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_basic_callbacks(
    out_dir="models",
    monitor="val_accuracy",
    patience_es=5,
    patience_lr=3,
    factor_lr=0.5,
    mode="max",
    checkpoint_name="best_model.h5"
):
    """
    Returns a list of basic callbacks:
     - ModelCheckpoint
     - EarlyStopping
     - ReduceLROnPlateau
     - CSVLogger

    Args:
        out_dir: Directory where model & logs will be saved
        monitor: Metric to monitor for checkpoint & early stopping
        patience_es: Early stopping patience
        patience_lr: Reduce LR patience
        factor_lr: LR reduction factor
        mode: "max" or "min"
        checkpoint_name: Filename for the best model

    Returns:
        list of tf.keras.callbacks
    """

    out_dir = ensure_dir(out_dir)
    log_file = out_dir / "training_log.csv"

    # Best model checkpoint
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / checkpoint_name),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        verbose=1
    )

    # Early stopping
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience_es,
        mode=mode,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce LR on plateau
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=factor_lr,
        patience=patience_lr,
        verbose=1
    )

    # CSV logger
    csv_logger = tf.keras.callbacks.CSVLogger(str(log_file), append=True)

    return [checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger]


def get_tensorboard_callback(log_dir="logs"):
    """
    Returns a TensorBoard callback.
    """
    log_dir = ensure_dir(log_dir)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1
    )


def get_all_callbacks(
    out_dir="models",
    use_tensorboard=False,
    **kwargs
):
    """
    Combines all callbacks into a single list.

    Args:
        out_dir: directory to save best model/logs
        use_tensorboard: whether to include TensorBoard callback
        **kwargs: passes extra arguments to get_basic_callbacks()

    Returns:
        list of callbacks
    """

    callbacks = get_basic_callbacks(out_dir=out_dir, **kwargs)

    if use_tensorboard:
        callbacks.append(get_tensorboard_callback(out_dir + "/tensorboard"))

    return callbacks


# -----------------------
# Example Usage (commented)
# -----------------------
'''
from src.training.callbacks import get_all_callbacks

callbacks = get_all_callbacks(
    out_dir="models/custom_cnn",
    monitor="val_accuracy",
    patience_es=6,
    patience_lr=3,
    use_tensorboard=True
)

model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
'''
