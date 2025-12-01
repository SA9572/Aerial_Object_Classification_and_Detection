"""
src/models/transfer_model.py

Transfer Learning model builders for Bird vs Drone classification.

Supported backbones:
    - "resnet50"
    - "mobilenetv2"
    - "efficientnetb0"

All models share:
    - Input -> Rescaling(1/255)  (0–255 → 0–1)
    - Data augmentation (flip, rotation, zoom)
    - Rescaling(0–1 → -1–1) before feeding to backbone
    - GlobalAveragePooling + Dense(256, ReLU) + Dropout
    - Final Dense(num_classes, softmax) for classification

IMPORTANT for low-RAM machines:
    By default, we DO NOT load ImageNet pretrained weights
    (pretrained_weights=None) to avoid MemoryError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50,
    MobileNetV2,
    EfficientNetB0,
)


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class TransferConfig:
    backbone: str = "resnet50"      # "resnet50" | "mobilenetv2" | "efficientnetb0"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 2
    dropout_rate: float = 0.3
    train_base: bool = False        # False: freeze backbone, True: fine-tune
    # ⚠️ Default: None → no pretrained weights (safe for low RAM)
    pretrained_weights: str | None = None


# ---------------------------------------------------------------------
# Helper: data augmentation block
# ---------------------------------------------------------------------
def build_data_augmentation() -> tf.keras.Sequential:
    """
    Returns a Keras Sequential with common image augmentations.
    Applied on the fly during training.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


# ---------------------------------------------------------------------
# Helper: choose backbone
# ---------------------------------------------------------------------
def _get_backbone(
    backbone: str,
    input_tensor: tf.Tensor,
    weights: str | None,
    train_base: bool,
) -> tf.keras.Model:
    """
    Internal function: create a backbone model (without top layers).

    NOTE:
        For your machine, you should pass weights=None.
        That means NO imagenet weights are loaded (prevents MemoryError).
    """
    backbone = backbone.lower()

    if backbone == "resnet50":
        base_model = ResNet50(
            include_top=False,
            weights=weights,       # usually None for you
            input_tensor=input_tensor,
        )
    elif backbone == "mobilenetv2":
        base_model = MobileNetV2(
            include_top=False,
            weights=weights,
            input_tensor=input_tensor,
        )
    elif backbone == "efficientnetb0":
        base_model = EfficientNetB0(
            include_top=False,
            weights=weights,
            input_tensor=input_tensor,
        )
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported: 'resnet50', 'mobilenetv2', 'efficientnetb0'."
        )

    base_model.trainable = train_base
    return base_model


# ---------------------------------------------------------------------
# Public API: build_transfer_model
# ---------------------------------------------------------------------
def build_transfer_model(
    backbone: str = "resnet50",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    train_base: bool = False,
    # ⚠️ Default is None → NO imagenet weights, safe on low-RAM
    pretrained_weights: str | None = None,
) -> tf.keras.Model:
    """
    Build a transfer-learning classifier with the given backbone.

    Pipeline:
        inputs (0–255) ->
        Rescale(1/255) (→ 0–1) ->
        Data Augmentation ->
        Rescale(2, offset=-1) (→ -1–1) ->
        Backbone (ResNet50 / MobileNetV2 / EfficientNetB0, no top) ->
        GlobalAveragePooling ->
        Dense(256, ReLU) + Dropout ->
        Dense(num_classes, Softmax)

    Args:
        backbone: "resnet50", "mobilenetv2", or "efficientnetb0"
        input_shape: (H, W, C)
        num_classes: number of output classes
        dropout_rate: dropout before final classification layer
        train_base: if True, backbone is trainable (fine-tuning)
        pretrained_weights: None (recommended for low RAM) or "imagenet"

    Returns:
        tf.keras.Model (uncompiled).
    """
    backbone = backbone.lower()

    # Input layer: expect 0–255 images
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 1) 0–255 -> 0–1
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)

    # 2) Data augmentation
    aug = build_data_augmentation()
    x = aug(x)

    # 3) 0–1 -> -1–1
    x = layers.Rescaling(2.0, offset=-1.0, name="rescale_minus1_1")(x)

    # 4) Backbone without top
    base_model = _get_backbone(
        backbone=backbone,
        input_tensor=x,
        weights=pretrained_weights,   # will be None in your case
        train_base=train_base,
    )
    x = base_model.output

    # 5) Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(
        256,
        activation="relu",
        name="head_dense_256",
    )(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_2")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model_name = f"{backbone.upper()}_classifier"
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ---------------------------------------------------------------------
# Convenience wrappers for specific backbones (optional)
# ---------------------------------------------------------------------
def build_resnet50_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    train_base: bool = False,
    pretrained_weights: str | None = None,
) -> tf.keras.Model:
    return build_transfer_model(
        backbone="resnet50",
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        train_base=train_base,
        pretrained_weights=pretrained_weights,
    )


def build_mobilenetv2_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    train_base: bool = False,
    pretrained_weights: str | None = None,
) -> tf.keras.Model:
    return build_transfer_model(
        backbone="mobilenetv2",
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        train_base=train_base,
        pretrained_weights=pretrained_weights,
    )


def build_efficientnetb0_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    train_base: bool = False,
    pretrained_weights: str | None = None,
) -> tf.keras.Model:
    return build_transfer_model(
        backbone="efficientnetb0",
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        train_base=train_base,
        pretrained_weights=pretrained_weights,
    )


# ---------------------------------------------------------------------
# CLI / Test entry point
# ---------------------------------------------------------------------
def _test_build_and_summary() -> None:
    """
    Build a default ResNet50 classifier and print the summary.

    NOTE:
        We use pretrained_weights=None here on purpose to avoid
        downloading / loading large ImageNet weights.
    """
    config = TransferConfig(
        backbone="resnet50",
        input_shape=(224, 224, 3),
        num_classes=2,
        dropout_rate=0.3,
        train_base=False,
        pretrained_weights=None,   # 👈 absolutely NO imagenet here
    )

    model = build_transfer_model(
        backbone=config.backbone,
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        train_base=config.train_base,
        pretrained_weights=config.pretrained_weights,
    )

    print("✅ Built Transfer Learning model with config:")
    print(config)
    print("\nModel summary:\n")
    model.summary()


if __name__ == "__main__":
    _test_build_and_summary()
