# """
# transfer_learning.py
# -----------------------
# Transfer learning model builder utilities.

# Provides:
#  - build_transfer_model(...) -> builds and compiles a Keras model using a pretrained backbone
#    (default: EfficientNetB0). The function supports freezing the base, unfreezing top layers
#    for fine-tuning, custom pooling & head, and returns a compiled model.

# Usage:
#     from src.models.transfer_learning import build_transfer_model

#     model = build_transfer_model(input_shape=(224,224,3),
#                                  num_classes=2,
#                                  backbone='EfficientNetB0',
#                                  base_trainable=False,
#                                  dropout_rate=0.3,
#                                  lr=1e-4)
# """

# from typing import Tuple, Optional
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers

# # Map simple names to keras applications
# _BACKBONES = {
#     "EfficientNetB0": tf.keras.applications.EfficientNetB0,
#     "MobileNetV2": tf.keras.applications.MobileNetV2,
#     "ResNet50": tf.keras.applications.ResNet50,
# }


# def _get_backbone(backbone_name: str, input_shape: Tuple[int, int, int], weights: Optional[str] = "imagenet", include_top: bool = False):
#     """
#     Return an instance of the chosen backbone model class (uncompiled).
#     """
#     name = backbone_name
#     if name not in _BACKBONES:
#         raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported: {list(_BACKBONES.keys())}")
#     BackboneClass = _BACKBONES[name]
#     # For ResNet50 we may want different default args; keep include_top=False to use our own head
#     base_model = BackboneClass(weights=weights, include_top=include_top, input_shape=input_shape)
#     return base_model


# def build_transfer_model(
#     input_shape: Tuple[int, int, int] = (224, 224, 3),
#     num_classes: int = 2,
#     backbone: str = "EfficientNetB0",
#     base_trainable: bool = False,
#     dropout_rate: float = 0.3,
#     pooling: str = "avg",  # 'avg' or 'max' or 'gap' (global average pooling)
#     lr: float = 1e-4,
#     metrics: Optional[list] = None,
# ):
#     """
#     Build and compile a transfer learning model.

#     Args:
#         input_shape: (H, W, C)
#         num_classes: number of target classes
#         backbone: one of 'EfficientNetB0', 'MobileNetV2', 'ResNet50'
#         base_trainable: whether to set the pretrained base as trainable
#         dropout_rate: dropout after dense layers
#         pooling: 'avg' or 'max' or 'gap' (global average pooling)
#         lr: learning rate for optimizer
#         metrics: list of keras metrics (defaults to accuracy, precision, recall)

#     Returns:
#         compiled tf.keras.Model
#     """
#     if metrics is None:
#         metrics = ["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]

#     # Input
#     inputs = layers.Input(shape=input_shape, name="input_image")

#     # Preprocessing: use appropriate application preprocessing if available
#     # EfficientNetB0 expects inputs in [0,255] and has its own preprocess_input. However in most pipelines we normalize to [0,1]
#     # Choose to apply Rescaling here (consistent with dataloader) and let fine-tuning handle specifics if needed
#     x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

#     # Backbone (pretrained)
#     base_model = _get_backbone(backbone, input_shape=input_shape, weights="imagenet", include_top=False)
#     base_model.trainable = base_trainable

#     # Connect backbone
#     x = base_model(x, training=False)

#     # Pooling
#     if pooling == "avg" or pooling == "gap":
#         x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
#     elif pooling == "max":
#         x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
#     else:
#         # fallback to gap
#         x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

#     # Head
#     x = layers.Dense(256, activation="relu", name="head_dense")(x)
#     x = layers.BatchNormalization(name="head_bn")(x)
#     x = layers.Dropout(dropout_rate, name="head_dropout")(x)

#     # Output activation & loss choice
#     if num_classes == 1:
#         activation = "sigmoid"
#         loss = "binary_crossentropy"
#     else:
#         activation = "softmax"
#         loss = "sparse_categorical_crossentropy"

#     outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

#     model = models.Model(inputs=inputs, outputs=outputs, name=f"tl_{backbone}")

#     # Compile
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=lr),
#         loss=loss,
#         metrics=metrics,
#     )

#     return model


# # Optional helper to unfreeze top N layers of the backbone for fine-tuning
# def unfreeze_backbone_top_layers(model: tf.keras.Model, backbone_layer_name_substring: str = None, unfreeze_layers: int = 20):
#     """
#     Unfreeze the top `unfreeze_layers` layers of the backbone within `model`.
#     The function tries to locate the backbone by searching for a layer whose name contains backbone_layer_name_substring.
#     If backbone_layer_name_substring is None, it will search for the first layer that has weights and is not part of the head.

#     Args:
#         model: compiled Keras model returned by build_transfer_model
#         backbone_layer_name_substring: substring to search for to identify backbone root (e.g., 'efficientnetb0')
#         unfreeze_layers: number of layers from the top to unfreeze

#     Returns:
#         None (mutates model in-place)
#     """
#     # Find backbone layers (heuristic)
#     # Search layers in reverse and unfreeze the first `unfreeze_layers` encountered that are trainable=False
#     # Note: this does not change compile state; user should recompile model after changing trainable flags.
#     count = 0
#     for layer in reversed(model.layers):
#         # Skip head layers which typically include 'predictions' or 'head_dense'
#         if layer.name in ("predictions", "head_dense", "head_bn", "head_dropout", "global_avg_pool", "global_max_pool"):
#             continue
#         # Only consider layers that have weights (i.e., part of backbone)
#         if layer.weights:
#             layer.trainable = True
#             count += 1
#         if count >= unfreeze_layers:
#             break
#     # Note: need to recompile model externally after calling this helper (with a lower lr).
"""
transfer_learning.py
-----------------------
Transfer learning model builder utilities.

Provides:
 - build_transfer_model(...) -> builds and compiles a Keras model using a pretrained backbone
   (default: EfficientNetB0). Supports freezing the base, unfreezing top layers for fine-tuning,
   custom pooling & head, and returns a compiled model.

Notes:
 - This builder assumes input images are scaled to [0,1]. If you use application-specific
   preprocess_input (e.g. EfficientNet preprocess), handle it consistently in your pipeline.
"""

from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Map simple names to keras applications
_BACKBONES = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet50": tf.keras.applications.ResNet50,
}


def _get_backbone(backbone_name: str, input_shape: Tuple[int, int, int], weights: Optional[str] = "imagenet", include_top: bool = False):
    """
    Return an instance of the chosen backbone model class (uncompiled).
    """
    if backbone_name not in _BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported: {list(_BACKBONES.keys())}")
    BackboneClass = _BACKBONES[backbone_name]
    base_model = BackboneClass(weights=weights, include_top=include_top, input_shape=input_shape)
    return base_model


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    backbone: str = "EfficientNetB0",
    base_trainable: bool = False,
    fine_tune_at: Optional[int] = None,
    dropout_rate: float = 0.3,
    pooling: str = "avg",  # 'avg' or 'max'
    lr: float = 1e-4,
    metrics: Optional[List] = None,
) -> tf.keras.Model:
    """
    Build and compile a transfer learning model.

    Args:
        input_shape: (H, W, C)
        num_classes: number of target classes
        backbone: one of 'EfficientNetB0', 'MobileNetV2', 'ResNet50'
        base_trainable: whether to set the pretrained base as trainable initially
        fine_tune_at: if int, unfreeze the top `fine_tune_at` layers of the backbone after loading
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

    # consistent rescaling (match DataLoader which outputs [0,1])
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Backbone
    base_model = _get_backbone(backbone, input_shape=input_shape, weights="imagenet", include_top=False)
    base_model.trainable = base_trainable

    # Connect backbone (ensure running=False for frozen base during training)
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

    # Output activation & loss
    if num_classes == 1:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = "sparse_categorical_crossentropy"

    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"tl_{backbone}")

    # Optionally unfreeze top N layers of base (do this before compile)
    if fine_tune_at is not None and fine_tune_at > 0:
        # ensure base_model is trainable first
        base_model.trainable = True
        # count layers with weights and unfreeze top fine_tune_at of them
        count = 0
        for layer in reversed(base_model.layers):
            if layer.weights:
                layer.trainable = True
                count += 1
            if count >= fine_tune_at:
                break

    # Compile with safe multiclass metrics
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )

    return model
