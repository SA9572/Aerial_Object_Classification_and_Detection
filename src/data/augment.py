"""
src/data/augment.py

Offline data augmentation for the classification dataset.

Reads images from:

    data/processed/
        train/
          bird/
          drone/
        valid/
          bird/
          drone/
        test/
          bird/
          drone/

and writes augmented images to:

    data/augmented/
        train/
          bird/
          drone/
        valid/
          bird/
          drone/
        test/
          bird/
          drone/

Each original image can generate N augmented variants, saved with
a suffix like: originalname_aug1.jpg, originalname_aug2.jpg, ...

Usage (from project root):

    python -m src.data.augment

Options:

    --root        Project root path (optional, auto-detected if not given)
    --num-aug     Number of augmented images per original image (default: 2)
    --splits      Splits to augment (default: train only)
                   Example: --splits train valid
"""

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------
# Path utilities (similar style to preprocess.py)
# ---------------------------------------------------------------------
def get_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path().resolve()
    if start.name == "notebooks":
        return start.parent
    return start


def get_processed_and_augmented_paths(project_root: Path | None = None) -> Tuple[Path, Path]:
    """
    Returns:
        processed_dir: data/processed
        augmented_dir: data/augmented
    """
    if project_root is None:
        project_root = get_project_root()

    processed_dir = project_root / "data" / "processed"
    augmented_dir = project_root / "data" / "augmented"

    return processed_dir, augmented_dir


def iter_image_files(directory: Path) -> Iterable[Path]:
    """
    Yield all image files in a directory with allowed extensions.
    Does NOT recurse into subdirectories.
    """
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


# ---------------------------------------------------------------------
# Augmentation operations
# ---------------------------------------------------------------------
def random_flip(img: Image.Image) -> Image.Image:
    r = random.random()
    if r < 0.33:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif r < 0.66:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return img


def random_rotate(img: Image.Image, max_angle: int = 20) -> Image.Image:
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR)


def random_brightness(img: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)


def random_contrast(img: Image.Image, factor_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)


def random_zoom_crop(img: Image.Image, max_crop: float = 0.1) -> Image.Image:
    """
    Randomly crop a small border and resize back to original size (simulates zoom).
    max_crop is the maximum proportion to crop from each side.
    """
    w, h = img.size
    dw = int(w * max_crop)
    dh = int(h * max_crop)

    left = random.randint(0, dw)
    top = random.randint(0, dh)
    right = random.randint(w - dw, w)
    bottom = random.randint(h - dh, h)

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize((w, h), Image.BILINEAR)


def random_noise(img: Image.Image, noise_strength: float = 0.03) -> Image.Image:
    """
    Add small Gaussian noise. Work in numpy and clip back to [0,255].
    noise_strength is std-dev as a fraction of 255.
    """
    arr = np.array(img).astype(np.float32)
    noise = np.random.randn(*arr.shape) * (255.0 * noise_strength)
    arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_noisy)


def apply_random_augmentations(img: Image.Image) -> Image.Image:
    """
    Apply a random combination of augmentations.
    Each call should produce a slightly different image.
    """
    # Random flip
    img = random_flip(img)

    # Random rotation
    if random.random() < 0.8:
        img = random_rotate(img, max_angle=20)

    # Random brightness & contrast
    if random.random() < 0.8:
        img = random_brightness(img, (0.7, 1.3))
    if random.random() < 0.8:
        img = random_contrast(img, (0.8, 1.2))

    # Random zoom crop
    if random.random() < 0.7:
        img = random_zoom_crop(img, max_crop=0.10)

    # Random noise
    if random.random() < 0.5:
        img = random_noise(img, noise_strength=0.02)

    # Occasionally convert to grayscale and back (simulate different cameras)
    if random.random() < 0.2:
        img = ImageOps.grayscale(img).convert("RGB")

    return img


# ---------------------------------------------------------------------
# Main augmentation loop
# ---------------------------------------------------------------------
def augment_split_class(
    src_dir: Path,
    dst_dir: Path,
    num_aug_per_image: int,
) -> None:
    """
    For all images in src_dir, create N augmented copies
    in dst_dir with suffix _augX.
    """
    if not src_dir.exists():
        print(f"[WARN] Source folder missing, skipping: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(iter_image_files(src_dir))
    if not img_files:
        print(f"[WARN] No images found in: {src_dir}")
        return

    print(f"\nAugmenting images from: {src_dir}")
    print(f"Saving augmented images to: {dst_dir}")
    print(f"Found {len(img_files)} original images. Generating {num_aug_per_image} aug per image.")

    for img_path in tqdm(img_files, desc=str(src_dir.name), unit="img"):
        try:
            orig = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {img_path}: {e}")
            continue

        stem = img_path.stem
        suffix = img_path.suffix.lower()

        for i in range(1, num_aug_per_image + 1):
            aug_img = apply_random_augmentations(orig)
            aug_name = f"{stem}_aug{i}{suffix}"
            aug_path = dst_dir / aug_name
            aug_img.save(aug_path)


def augment_dataset(
    processed_root: Path,
    augmented_root: Path,
    num_aug_per_image: int = 2,
    splits: Iterable[str] = ("train",),
    classes: Iterable[str] = ("bird", "drone"),
) -> None:
    """
    Offline augment dataset from processed_root to augmented_root.

    Args:
        processed_root: data/processed
        augmented_root: data/augmented
        num_aug_per_image: how many augmented images to create per original
        splits: which splits to augment (e.g. ("train",) or ("train","valid"))
        classes: which classes to augment
    """
    print("Processed dataset input :", processed_root)
    print("Augmented dataset output:", augmented_root)
    print("Splits to augment       :", splits)
    print("Classes                 :", classes)
    print("Aug per original image  :", num_aug_per_image)
    print("-" * 60)

    if not processed_root.exists():
        raise FileNotFoundError(f"Processed dataset folder does not exist: {processed_root}")

    for split in splits:
        for cls in classes:
            src_dir = processed_root / split / cls
            dst_dir = augmented_root / split / cls
            augment_split_class(src_dir, dst_dir, num_aug_per_image)

    print("\n✅ Augmentation completed.")
    print(f"Augmented images saved under: {augmented_root}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline data augmentation for classification dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root path. If not provided, auto-detect from current directory.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=2,
        help="Number of augmented images to create per original image.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to augment (space separated). Example: --splits train valid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(args.root).resolve() if args.root else get_project_root()
    processed_root, augmented_root = get_processed_and_augmented_paths(project_root)

    augment_dataset(
        processed_root=processed_root,
        augmented_root=augmented_root,
        num_aug_per_image=args.num_aug,
        splits=tuple(args.splits),
        classes=("bird", "drone"),
    )


if __name__ == "__main__":
    main()
