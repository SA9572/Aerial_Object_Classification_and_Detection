"""
augmentation.py
-----------------------
Albumentations-based augmentation utilities for the project.

Provides:
 - get_train_augmentations(img_size)
 - get_valid_augmentations(img_size)
 - augment_image_np(image_np, aug)         # apply to a numpy HxWxC image (uint8 or float)
 - augment_image_tf(image_tensor, aug, normalize_out=True)
     -> wrapper so you can use augmentation inside a tf.data pipeline via .map()

Notes:
 - Albumentations expects images in HWC order and uint8 (0-255). The helpers handle
   conversion from float [0,1] to uint8 and back.
 - To use in tf.data, call:
     train_ds = train_ds.map(lambda x,y: (augment_image_tf(x, my_aug), y), num_parallel_calls=...)
   where `x` is a batch or single image tensor. The helper supports both single images
   and batches (it will apply augmentation per image).
"""

from typing import Tuple
import numpy as np
from PIL import Image
import albumentations as A
import tensorflow as tf


def get_train_augmentations(img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Returns an Albumentations Compose object for training augmentation.

    Args:
        img_size: (height, width) target output size

    Returns:
        albumentations.Compose
    """
    h, w = img_size
    aug = A.Compose(
        [
            A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0)), A.MotionBlur(blur_limit=3)], p=0.3),
            # final safety resize to exact size
            A.Resize(height=h, width=w, interpolation=1, p=1.0),
        ],
        p=1.0,
    )
    return aug


def get_valid_augmentations(img_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Returns a minimal augmentation (deterministic) for validation / test.

    Args:
        img_size: (height, width)

    Returns:
        albumentations.Compose
    """
    h, w = img_size
    aug = A.Compose(
        [
            A.Resize(height=h, width=w, interpolation=1, p=1.0),
        ],
        p=1.0,
    )
    return aug


# --------------------------
# Numpy-based application
# --------------------------
def augment_image_np(image: np.ndarray, aug: A.Compose) -> np.ndarray:
    """
    Apply augmentation to a single image in numpy format.

    Args:
        image: H x W x C numpy array. dtype can be uint8 or float32 ([0,1]).
        aug: Albumentations Compose

    Returns:
        Augmented image as uint8 numpy array (H x W x C) with values in 0-255.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array (H,W,C)")

    # If float in [0,1], convert to uint8 0-255
    if image.dtype == np.float32 or image.dtype == np.float64:
        # clip then scale
        img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    # Albumentations expects HWC uint8
    augmented = aug(image=img)["image"]
    return augmented


# --------------------------
# TF wrapper for tf.data
# --------------------------
def _augment_single_image_for_tf(img_np, aug) -> np.ndarray:
    """
    Helper used inside tf.numpy_function. Accepts a single image numpy array (H,W,C) float32 [0,1]
    or uint8 and returns augmented image as float32 [0,1].
    """
    # ensure numpy array
    img = np.array(img_np)

    # handle batched tensors passed accidentally (take first)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # convert floats to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)

    augmented = aug(image=img_uint8)["image"]

    # convert back to float32 [0,1]
    augmented = augmented.astype(np.float32) / 255.0
    return augmented


def augment_image_tf(image_tensor, aug: A.Compose, normalize_out: bool = True):
    """
    TensorFlow-friendly augmentation function to be used inside tf.data pipelines.

    Usage (single-image pipelines):
        train_ds = train_ds.map(lambda x,y: (augment_image_tf(x, train_aug), y),
                                 num_parallel_calls=tf.data.AUTOTUNE)

    Args:
        image_tensor: tf.Tensor shape (H,W,C) or (B,H,W,C) with dtype float32 (0-1) or uint8 (0-255)
        aug: Albumentations Compose object
        normalize_out: if True, output will be float32 in range [0,1]. Otherwise uint8 0-255 as tf.float32.

    Returns:
        Augmented image tensor (same shape as single input): tf.Tensor dtype float32
    """
    # We need to wrap the numpy augmentation function.
    def _py_aug(img_np):
        # if batched, apply per image and stack
        img_np = np.array(img_np)
        if img_np.ndim == 4:  # batch
            out_list = []
            for i in range(img_np.shape[0]):
                out = _augment_single_image_for_tf(img_np[i], aug)
                out_list.append(out)
            out_arr = np.stack(out_list, axis=0)
        else:
            out_arr = _augment_single_image_for_tf(img_np, aug)
        if normalize_out:
            return out_arr.astype(np.float32)
        else:
            # return as float32 but in 0-255 range
            return (out_arr * 255.0).astype(np.float32)

    # Use tf.numpy_function to call _py_aug
    augmented = tf.numpy_function(func=_py_aug, inp=[image_tensor], Tout=tf.float32)

    # Set shape information if possible (unknown dimension may appear)
    # Try to infer shape from input
    try:
        input_shape = image_tensor.shape
        augmented.set_shape(input_shape)
    except Exception:
        # fallback: set (None, None, None, 3) for batches or (None, None, 3)
        if image_tensor.shape.ndims == 4:
            augmented.set_shape([None, None, None, 3])
        else:
            augmented.set_shape([None, None, 3])

    return augmented


# --------------------------
# Utility: save augmented sample
# --------------------------
def save_augmented_sample(image_np: np.ndarray, aug: A.Compose, out_path: str):
    """
    Apply augmentation and save a sample to disk (useful for debugging).

    Args:
        image_np: HWC numpy array (uint8 or float32)
        aug: Albumentations Compose
        out_path: file path to save (e.g. 'outputs/aug_sample.jpg')
    """
    aug_img = augment_image_np(image_np, aug)
    im = Image.fromarray(aug_img)
    im.save(out_path)


# --------------------------
# Example usage (in comments)
# --------------------------
# from src.data.augmentation import get_train_augmentations, augment_image_tf
# import tensorflow as tf
#
# train_aug = get_train_augmentations((224,224))
#
# # If your tf.data pipeline produces float images in [0,1]:
# train_ds = train_ds.map(lambda x,y: (augment_image_tf(x, train_aug), y),
#                         num_parallel_calls=tf.data.AUTOTUNE)
#
# # For online debugging / saving:
# import cv2
# img = cv2.imread("data/classification_dataset/train/bird/0001.jpg")  # BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# aug = get_train_augmentations((224,224))
# from src.data.augmentation import augment_image_np
# out = augment_image_np(img, aug)
# from PIL import Image
# Image.fromarray(out).save("outputs/aug_debug.jpg")
