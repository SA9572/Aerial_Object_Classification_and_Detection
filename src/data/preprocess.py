"""
src/data/preprocess.py

Utility functions for preprocessing the classification dataset:

- Resize images (e.g. to 224x224)
- Normalize format (RGB)
- Save to data/processed with the same split/class structure.

Usage (from project root):

    python -m src.data.preprocess

or

    python src/data/preprocess.py

The script assumes this structure:

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

Output (created automatically):

    data/
      processed/
        train/
          bird/
          drone/
        valid/
          bird/
          drone/
        test/
          bird/
          drone/
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image
from tqdm import tqdm

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    """
    Try to detect the project root.
    If you're inside 'notebooks', it will go one level up.
    Otherwise it assumes current working directory is the root.
    """
    if start is None:
        start = Path().resolve()

    if start.name == "notebooks":
        return start.parent

    return start


def get_classification_paths(
    project_root: Path | None = None,
) -> Tuple[Path, Path]:
    """
    Returns:
        raw_dir:      data/classification_dataset
        processed_dir:data/processed
    """
    if project_root is None:
        project_root = get_project_root()

    raw_dir = project_root / "data" / "classification_dataset"
    processed_dir = project_root / "data" / "processed"

    return raw_dir, processed_dir


# ---------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------
def iter_image_files(
    directory: Path,
    exts: Iterable[str] = IMAGE_EXTS,
) -> Iterable[Path]:
    """
    Yield all image files in a directory with allowed extensions.
    Does NOT recurse into subdirectories.
    """
    exts = {e.lower() for e in exts}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def resize_and_save_image(
    src_path: Path,
    dst_path: Path,
    size: Tuple[int, int] = (224, 224),
) -> None:
    """
    Open an image, convert to RGB, resize, and save to dst_path.
    """
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(size)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
    except Exception as e:
        print(f"[WARN] Failed to process image {src_path}: {e}")


def preprocess_classification_dataset(
    raw_root: Path,
    processed_root: Path,
    image_size: Tuple[int, int] = (224, 224),
    splits: Iterable[str] = ("train", "valid", "test"),
    classes: Iterable[str] = ("bird", "drone"),
    overwrite: bool = False,
) -> None:
    """
    Resize all classification images and save them to processed_root.

    Args:
        raw_root:       Path to data/classification_dataset
        processed_root: Path to data/processed
        image_size:     (width, height) for resizing
        splits:         Dataset splits to process
        classes:        Class subfolders to process
        overwrite:      If False, skip files that already exist in processed_root
    """
    print("Raw classification dataset:", raw_root)
    print("Processed dataset output :", processed_root)
    print("Target image size        :", image_size)
    print("Splits                   :", splits)
    print("Classes                  :", classes)
    print("Overwrite existing?      :", overwrite)
    print("-" * 60)

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW dataset folder does not exist: {raw_root}")

    for split in splits:
        for cls in classes:
            src_dir = raw_root / split / cls
            dst_dir = processed_root / split / cls

            if not src_dir.exists():
                print(f"[WARN] Source folder missing, skipping: {src_dir}")
                continue

            img_files = list(iter_image_files(src_dir))
            if not img_files:
                print(f"[WARN] No images found in: {src_dir}")
                continue

            print(f"\nProcessing split='{split}', class='{cls}'")
            print(f"  Source : {src_dir}")
            print(f"  Target : {dst_dir}")
            print(f"  Images : {len(img_files)}")

            for img_path in tqdm(img_files, desc=f"{split}/{cls}", unit="img"):
                out_path = dst_dir / img_path.name
                if out_path.exists() and not overwrite:
                    # Skip if already processed
                    continue
                resize_and_save_image(img_path, out_path, size=image_size)

    print("\n✅ Preprocessing completed.")
    print(f"Processed images saved under: {processed_root}")


# ---------------------------------------------------------------------
# CLI (so you can run this file directly)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess classification dataset (resize & save)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detects from current directory.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Target image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Target image height.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already processed images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    raw_dir, processed_dir = get_classification_paths(project_root)

    preprocess_classification_dataset(
        raw_root=raw_dir,
        processed_root=processed_dir,
        image_size=(args.width, args.height),
        splits=("train", "valid", "test"),
        classes=("bird", "drone"),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
