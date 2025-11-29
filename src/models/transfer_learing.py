"""
transfer_learing.py  (yes, with the same filename spelling you already use)
------------------------------------------------
Transfer learning model builder utilities.

- Expects inputs normalized to [0,1] (DataLoader does /255.0)
- Internally rescales to [0,255] and applies the appropriate
  tf.keras.applications.<backbone>.preprocess_input
- Supports EfficientNetB0, MobileNetV2, ResNet50 as backbones
"""

from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Backbone classes
_BACKBONES = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet50": tf.keras.applications.ResNet50,
}

# Preprocess functions (for raw [0..255] images)
_PREPROCESS_FNS = {
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
}


def _get_backbone(
    backbone_name: str,
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    include_top: bool = False,
):
    """
    Return an instance of the chosen backbone model class (uncompiled).
    """
    if backbone_name not in _BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Supported: {list(_BACKBONES.keys())}"
        )
    BackboneClass = _BACKBONES[backbone_name]
    base_model = BackboneClass(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape,
    )
    return base_model


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    backbone: str = "EfficientNetB0",
    base_trainable: bool = False,
    dropout_rate: float = 0.3,
    pooling: str = "avg",  # 'avg' or 'max'
    lr: float = 1e-4,
    metrics: Optional[List] = None,
):
    """
    Build and compile a transfer learning model.

    Assumptions:
      - Input images are normalized to [0,1] before the model (DataLoader or preprocessing).
      - This function rescales to [0,255] and applies backbone-specific preprocess_input
        before feeding into the pretrained backbone.

    Args:
        input_shape: (H, W, C)
        num_classes: number of target classes
        backbone: one of 'EfficientNetB0', 'MobileNetV2', 'ResNet50'
        base_trainable: whether to set the pretrained base as trainable
        dropout_rate: dropout after dense layers
        pooling: 'avg' or 'max'
        lr: learning rate for optimizer
        metrics: list of keras metrics (defaults to SparseCategoricalAccuracy)

    Returns:
        compiled tf.keras.Model
    """
    if metrics is None:
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs  # expected [0,1]

    # Rescale back to [0,255] then apply backbone-specific preprocessing
    preprocess_fn = _PREPROCESS_FNS.get(backbone, None)
    if preprocess_fn is not None:
        def _pp(z):
            # z in [0,1] -> [0,255] -> preprocess_input
            return preprocess_fn(z * 255.0)

        x = layers.Lambda(_pp, name=f"{backbone}_preprocess")(x)

    # Backbone
    base_model = _get_backbone(
        backbone_name=backbone,
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
    )
    base_model.trainable = base_trainable

    x = base_model(x, training=False)

    # Pooling
    if pooling == "max":
        x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    else:
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Head
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Output
    if num_classes == 1:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = "sparse_categorical_crossentropy"

    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"tl_{backbone}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )

    return model


def unfreeze_backbone_top_layers(
    model: tf.keras.Model,
    unfreeze_layers: int = 20,
):
    """
    Unfreeze the top `unfreeze_layers` layers of the backbone within `model`.

    Note:
      - This mutates layer.trainable flags in place.
      - You must recompile the model with a lower learning rate afterwards.
    """
    count = 0
    for layer in reversed(model.layers):
        if layer.name in {
            "predictions",
            "head_dense",
            "head_bn",
            "head_dropout",
            "global_avg_pool",
            "global_max_pool",
        }:
            continue
        if layer.weights:
            layer.trainable = True
            count += 1
        if count >= unfreeze_layers:
            break
