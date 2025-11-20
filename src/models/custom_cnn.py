# """
# custom_cnn.py
# -----------------------
# Defines a clean, well-regularized custom CNN model for binary classification (bird vs drone).
# Provides a build_custom_cnn(...) function that returns a compiled Keras model.
# """

# from tensorflow.keras import layers, models, optimizers
# import tensorflow as tf

# def build_custom_cnn(input_shape=(224,224,3), num_classes=2, lr=1e-3):
#     """
#     Build and compile a custom CNN model.

#     Args:
#         input_shape (tuple): input image shape (H, W, C)
#         num_classes (int): number of output classes
#         lr (float): learning rate for the Adam optimizer

#     Returns:
#         compiled tf.keras.Model
#     """
#     inputs = layers.Input(shape=input_shape, name="input_image")

#     # Block 1
#     x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
#     x = layers.BatchNormalization(name="bn1")(x)
#     x = layers.MaxPooling2D(2, name="pool1")(x)

#     # Block 2
#     x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
#     x = layers.BatchNormalization(name="bn2")(x)
#     x = layers.MaxPooling2D(2, name="pool2")(x)

#     # Block 3
#     x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
#     x = layers.BatchNormalization(name="bn3")(x)
#     x = layers.MaxPooling2D(2, name="pool3")(x)
#     x = layers.Dropout(0.25, name="dropout3")(x)

#     # Block 4
#     x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
#     x = layers.BatchNormalization(name="bn4")(x)
#     x = layers.MaxPooling2D(2, name="pool4")(x)
#     x = layers.Dropout(0.3, name="dropout4")(x)

#     # Head
#     x = layers.GlobalAveragePooling2D(name="gap")(x)
#     x = layers.Dense(256, activation="relu", name="dense1")(x)
#     x = layers.BatchNormalization(name="bn_head")(x)
#     x = layers.Dropout(0.4, name="dropout_head")(x)

#     if num_classes == 1:
#         activation = "sigmoid"
#         loss = "binary_crossentropy"
#     else:
#         activation = "softmax"
#         loss = "sparse_categorical_crossentropy"

#     outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

#     model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn")

#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=lr),
#         loss=loss,
#         metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
#     )

#     return model
"""
custom_cnn.py
-----------------------
Defines a well-regularized custom CNN model for classification.
Compiled using SparseCategoricalAccuracy so it's safe for multi-class training.
"""

from tensorflow.keras import layers, models, optimizers
import tensorflow as tf


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2, lr=1e-3, metrics=None):
    """
    Build and compile a custom CNN model.

    Args:
        input_shape: (H, W, C)
        num_classes: number of output classes (2 for bird/drone)
        lr: learning rate
        metrics: list of tf.keras metrics (optional). If None, uses SparseCategoricalAccuracy.

    Returns:
        compiled tf.keras.Model
    """
    if metrics is None:
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.25, name="dropout3")(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)
    x = layers.Dropout(0.3, name="dropout4")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.4, name="dropout_head")(x)

    # Choose activation + loss by num_classes
    if num_classes == 1:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = "sparse_categorical_crossentropy"

    outputs = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics
    )

    return model
