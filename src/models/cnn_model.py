"""
src/models/cnn_model.py

Defines a custom CNN model for Bird vs Drone classification.

This module provides:
    - build_custom_cnn(...) : returns a Keras Model

Default setup:
    - Input shape: (224, 224, 3)
    - num_classes: 2  (bird, drone)
    - Convolutional blocks + BatchNorm + Dropout
    - Final softmax layer for multi-class classification

Typical usage (in training code):

    from src.models.cnn_model import build_custom_cnn

    model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    model.fit(...)

You can also run this file directly to see the architecture:

    python -m src.models.cnn_model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ---------------------------------------------------------------------
# Configuration dataclass (optional, for flexibility)
# ---------------------------------------------------------------------
@dataclass
class CNNConfig:
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    base_filters: int = 32
    l2_reg: float = 1e-4
    dropout_rate: float = 0.5


# ---------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------
def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    pool: bool = True,
    l2_reg: float = 1e-4,
    name_prefix: str | None = None,
) -> tf.Tensor:
    """
    A small convenience block:
        Conv2D -> BatchNorm -> ReLU -> (MaxPool)

    Args:
        x: input tensor
        filters: number of filters
        kernel_size: conv kernel size
        pool: whether to apply MaxPooling2D
        l2_reg: L2 regularization factor
        name_prefix: optional prefix for layer names
    """
    if name_prefix is None:
        name_prefix = f"conv_{filters}"

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)

    if pool:
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(x)

    return x


# ---------------------------------------------------------------------
# Main model builder
# ---------------------------------------------------------------------
def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    base_filters: int = 32,
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """
    Build a custom CNN model for image classification.

    Args:
        input_shape: (H, W, C), e.g. (224, 224, 3)
        num_classes: number of output classes (2 for bird/drone)
        base_filters: number of filters in the first conv block
        l2_reg: L2 regularization factor
        dropout_rate: dropout rate before the final dense layer

    Returns:
        A tf.keras.Model instance (uncompiled).
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Normalize inputs to [0, 1] if they are in [0, 255]
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # Convolutional feature extractor
    x = conv_block(x, base_filters, name_prefix="block1", l2_reg=l2_reg)
    x = conv_block(x, base_filters * 2, name_prefix="block2", l2_reg=l2_reg)
    x = conv_block(x, base_filters * 4, name_prefix="block3", l2_reg=l2_reg)
    x = conv_block(x, base_filters * 8, name_prefix="block4", l2_reg=l2_reg)

    # Global feature aggregation
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Output layer
    if num_classes == 1:
        # Binary classification with a single logit
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        # Multi-class classification with softmax
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    return model


# ---------------------------------------------------------------------
# CLI / Test entry point
# ---------------------------------------------------------------------
def _test_build_and_summary() -> None:
    """
    Simple test function:
      - Build a default model
      - Print summary
    """
    config = CNNConfig()  # default values
    model = build_custom_cnn(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        base_filters=config.base_filters,
        l2_reg=config.l2_reg,
        dropout_rate=config.dropout_rate,
    )

    print("✅ Built Custom CNN model with config:")
    print(config)
    print("\nModel summary:\n")
    model.summary()


if __name__ == "__main__":
    _test_build_and_summary()
