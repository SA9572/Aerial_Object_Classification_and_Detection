"""
dataloader.py
-----------------------
Handles loading of image datasets for training, validation, and testing.
Creates tf.data pipelines with batching, caching, normalization, and prefetch.
Used by both Custom CNN and Transfer Learning models.
"""

import os
from pathlib import Path
import tensorflow as tf


class DataLoader:
    def __init__(
        self,
        data_dir,
        img_size=(224, 224),
        batch_size=32,
        seed=42,
        normalize=True,
    ):
        """
        Initialize the dataloader.

        Args:
            data_dir (str or Path): Path to classification_dataset folder
            img_size (tuple): Image resize (height, width)
            batch_size (int): Batch size for tf.data
            seed (int): Random seed for reproducibility
            normalize (bool): Apply normalization 1./255
        """

        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.normalize = normalize

        # Normalization layer
        self.normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

        # tf.data autotune
        self.AUTOTUNE = tf.data.AUTOTUNE

    def _load_split(self, split_name, shuffle=False):
        """
        Internal method to load a dataset split (train/valid/test)
        """
        split_path = self.data_dir / split_name

        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} does not exist")

        ds = tf.keras.preprocessing.image_dataset_from_directory(
            split_path,
            labels="inferred",
            label_mode="int",
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
        )

        # Extract class names
        self.class_names = ds.class_names

        # Normalize
        if self.normalize:
            ds = ds.map(lambda x, y: (self.normalization_layer(x), y),
                        num_parallel_calls=self.AUTOTUNE)

        # Optimize pipeline
        ds = ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def load(self):
        """
        Loads train, valid, test datasets.
        Returns:
            train_ds, val_ds, test_ds, class_names
        """

        print("🔁 Loading TRAIN dataset...")
        train_ds = self._load_split("train", shuffle=True)

        print("🔁 Loading VALID dataset...")
        val_ds = self._load_split("valid", shuffle=False)

        print("🔁 Loading TEST dataset...")
        test_ds = self._load_split("test", shuffle=False)

        print(f"📌 Classes found: {self.class_names}")

        return train_ds, val_ds, test_ds, self.class_names

    def count_images(self):
        """
        Count images in train/valid/test for each class.
        Returns:
            dict with counts
        """
        counts = {}
        for split in ["train", "valid", "test"]:
            split_path = self.data_dir / split
            split_counts = {}
            for cls in os.listdir(split_path):
                cls_path = split_path / cls
                if cls_path.is_dir():
                    total = len(list(cls_path.glob("*.jpg"))) + \
                            len(list(cls_path.glob("*.jpeg"))) + \
                            len(list(cls_path.glob("*.png")))
                    split_counts[cls] = total
            counts[split] = split_counts
        return counts
