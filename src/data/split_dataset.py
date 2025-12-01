"""
src/data/split_dataset.py

Utility to split a flat classification dataset into train/valid/test folders.

Expected source structure:

    data/
      raw_classification/
        bird/
          img001.jpg
          img002.jpg
          ...
        drone/
          img101.jpg
          img102.jpg
          ...

This script will create:

    data/
      classification_dataset/
        train/
          bird/
          drone/
        valid/
          bird/
          drone/
        test/
          bird/
          drone/

You can control:
  - train/valid/test ratios
  - whether to COPY or MOVE the files
  - which classes and splits to use

Typical usage (from project root):

    python -m src.data.split_dataset

or:

    python -m src.data.split_dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

If you already have data/classification_dataset filled, you DO NOT need this script.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def get_source_and_target_paths(
    project_root: Path | None = None,
    source_dir_name: str = "raw_classification",
    target_dir_name: str = "classification_dataset",
) -> Tuple[Path, Path]:
    """
    Returns:
        src_root: data/raw_classification
        dst_root: data/classification_dataset
    """
    if project_root is None:
        project_root = get_project_root()
    data_root = project_root / "data"

    src_root = data_root / source_dir_name
    dst_root = data_root / target_dir_name

    return src_root, dst_root


def list_class_folders(src_root: Path) -> List[Path]:
    """
    Returns a list of class folders (e.g. bird, drone) directly under src_root.
    """
    class_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    return class_dirs


def list_images_in_folder(folder: Path) -> List[Path]:
    """
    Returns a sorted list of image files in a folder.
    """
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    files.sort()
    return files


# ---------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------
def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split n indices into train/val/test based on ratios.
    Returns dict with keys 'train', 'valid', 'test'.
    """
    if not (0 < train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test ratios must sum to 1.0 (got {total}).")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remaining go to test
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": train_idx,
        "valid": val_idx,
        "test": test_idx,
    }


def copy_or_move_file(src: Path, dst: Path, move: bool = False) -> None:
    """
    Copy or move a file from src to dst.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def split_class_folder(
    class_dir: Path,
    dst_root: Path,
    splits: Dict[str, List[int]],
    files: List[Path],
    move: bool = False,
) -> None:
    """
    Given a class folder and file list, copy/move files into
    dst_root / split / class_name.
    """
    class_name = class_dir.name
    print(f"\nClass: {class_name}")
    print(f"Total images: {len(files)}")

    for split_name, idx_list in splits.items():
        split_dir = dst_root / split_name / class_name
        print(f"  -> {split_name}: {len(idx_list)} images -> {split_dir}")

        for idx in idx_list:
            src_file = files[idx]
            dst_file = split_dir / src_file.name
            copy_or_move_file(src_file, dst_file, move=move)


def split_dataset(
    src_root: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    move: bool = False,
    seed: int = 42,
) -> None:
    """
    Split dataset from src_root into dst_root/train|valid|test.

    Args:
        src_root: e.g. data/raw_classification
        dst_root: e.g. data/classification_dataset
        train_ratio: fraction for train
        val_ratio: fraction for validation
        test_ratio: fraction for test
        move: if True, move files instead of copying
        seed: random seed for reproducible splits
    """
    print("Source root (raw):", src_root)
    print("Target root (split):", dst_root)
    print(f"Ratios -> train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
    print("Operation: ", "MOVE" if move else "COPY")
    print("Random seed:", seed)
    print("-" * 60)

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_root}")

    class_dirs = list_class_folders(src_root)
    if not class_dirs:
        raise ValueError(f"No class directories found under {src_root}")

    print("Found class folders:")
    for c in class_dirs:
        print("  -", c.name)

    for class_dir in class_dirs:
        files = list_images_in_folder(class_dir)
        n = len(files)
        if n == 0:
            print(f"[WARN] No images in class folder: {class_dir}")
            continue

        splits = split_indices(
            n=n,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_class_folder(class_dir, dst_root, splits, files, move=move)

    print("\n✅ Dataset split completed.")
    print(f"Split dataset is available at: {dst_root}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a raw classification dataset into train/valid/test folders."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    parser.add_argument(
        "--source-dir-name",
        type=str,
        default="raw_classification",
        help="Name of source folder under data/ (default: raw_classification).",
    )
    parser.add_argument(
        "--target-dir-name",
        type=str,
        default="classification_dataset",
        help="Name of target folder under data/ (default: classification_dataset).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    src_root, dst_root = get_source_and_target_paths(
        project_root,
        source_dir_name=args.source_dir_name,
        target_dir_name=args.target_dir_name,
    )

    split_dataset(
        src_root=src_root,
        dst_root=dst_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        move=args.move,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
