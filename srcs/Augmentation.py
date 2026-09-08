#!/usr/bin/env python3

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from image_utils import is_image_file


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def estimate_background_color(img: Image.Image) -> Tuple[int, int, int]:
    """Estimate a plausible fill color from the image corners."""
    pixels = np.asarray(img)
    patch_size = max(1, min(img.size) // 16)
    corners = np.concatenate(
        [
            pixels[:patch_size, :patch_size],
            pixels[:patch_size, -patch_size:],
            pixels[-patch_size:, :patch_size],
            pixels[-patch_size:, -patch_size:],
        ],
        axis=0,
    )
    median = np.median(corners.reshape(-1, 3), axis=0)
    return tuple(int(value) for value in median)


def flip_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    _ = rng
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def rotate_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-20.0, 20.0)
    return img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=estimate_background_color(img),
    )


def scaling_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    scale = rng.uniform(0.85, 1.15)
    if scale >= 1.0:
        new_w = int(width * scale)
        new_h = int(height * scale)
        scaled = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return scaled.crop((left, top, left + width, top + height))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    scaled = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height),
                       color=estimate_background_color(img))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas.paste(scaled, (left, top))
    return canvas


def illumination_augmentation(
    img: Image.Image,
    rng: random.Random,
) -> Image.Image:
    factor = rng.uniform(0.60, 1.40)
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.65, 1.45)
    return ImageEnhance.Contrast(img).enhance(factor)


def projective_augmentation(
    img: Image.Image,
    rng: random.Random,
) -> Image.Image:
    width, height = img.size
    margin_w = max(1, int(width * 0.06))
    margin_h = max(1, int(height * 0.06))
    src = [(0, 0), (width, 0), (width, height), (0, height)]
    dst = [
        (rng.randint(-margin_w, margin_w), rng.randint(-margin_h, margin_h)),
        (width + rng.randint(-margin_w, margin_w),
         rng.randint(-margin_h, margin_h)),
        (width + rng.randint(-margin_w, margin_w),
         height + rng.randint(-margin_h, margin_h)),
        (rng.randint(-margin_w, margin_w),
         height + rng.randint(-margin_h, margin_h)),
    ]
    coeffs = find_perspective_coeffs(dst, src)
    return img.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
        fillcolor=estimate_background_color(img),
    )


def find_perspective_coeffs(
    pa: List[Tuple[float, float]], pb: List[Tuple[float, float]]
) -> List[float]:
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append(
            [
                p1[0],
                p1[1],
                1,
                0,
                0,
                0,
                -p2[0] * p1[0],
                -p2[0] * p1[1],
            ]
        )
        matrix.append(
            [
                0,
                0,
                0,
                p1[0],
                p1[1],
                1,
                -p2[1] * p1[0],
                -p2[1] * p1[1],
            ]
        )
    a = np.array(matrix, dtype=np.float64)
    b = np.array(pb, dtype=np.float64).reshape(8)
    res = np.linalg.solve(a, b)
    return res.tolist()


@dataclass(frozen=True)
class Augmentation:
    name: str
    func: Callable[[Image.Image, random.Random], Image.Image]


AUGMENTATIONS: List[Augmentation] = [
    Augmentation("Flip", flip_augmentation),
    Augmentation("Rotate", rotate_augmentation),
    Augmentation("Scaling", scaling_augmentation),
    Augmentation("Illumination", illumination_augmentation),
    Augmentation("Contrast", contrast_augmentation),
    Augmentation("Projective", projective_augmentation),
]


def build_augmented_filename(
    path: Path,
    aug_name: str,
    index: Optional[int] = None,
) -> Path:
    stem = path.stem
    suffix = path.suffix if path.suffix else ".jpg"
    if index is None:
        new_name = f"{stem}_{aug_name}{suffix}"
    else:
        new_name = f"{stem}_{aug_name}_{index:04d}{suffix}"
    return path.with_name(new_name)


def is_augmented_filename(path: Path) -> bool:
    pattern = (
        r"_(Flip|Rotate|Scaling|Illumination|Contrast|Projective)"
        r"(_\d{4})?$"
    )
    return re.search(pattern, path.stem) is not None


def base_stem_without_augmentation(path: Path) -> str:
    pattern = (
        r"(?:_(Flip|Rotate|Scaling|Illumination|Contrast|Projective)"
        r"(_\d{4})?)+$"
    )
    return re.sub(pattern, "", path.stem)


def generate_six_for_image(
    image_path: Path,
    output_dir: Optional[Path],
    seed: int,
    show: bool,
) -> Dict[str, Path]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_image_file(image_path):
        raise ValueError(f"Unsupported file: {image_path}")
    rng = random.Random(seed)
    image = ensure_rgb(Image.open(image_path))
    destination_root = output_dir if output_dir else image_path.parent
    destination_root.mkdir(parents=True, exist_ok=True)
    created_files: Dict[str, Path] = {}
    augmented_images: Dict[str, Image.Image] = {}
    for aug in AUGMENTATIONS:
        transformed = aug.func(image, rng)
        target_path = build_augmented_filename(
            destination_root / image_path.name, aug.name)
        transformed.save(target_path)
        created_files[aug.name] = target_path
        augmented_images[aug.name] = transformed
    if show:
        display_augmentations(image, augmented_images, image_path.name)
    return created_files


def display_augmentations(
    original: Image.Image,
    augmented_images: Dict[str, Image.Image],
    title_base: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for idx, aug in enumerate(AUGMENTATIONS, start=1):
        axes[idx].imshow(augmented_images[aug.name])
        axes[idx].set_title(aug.name)
        axes[idx].axis("off")
    for idx in range(len(AUGMENTATIONS) + 1, len(axes)):
        axes[idx].axis("off")
    fig.suptitle(f"Data augmentation - {title_base}", fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def list_images(directory: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and is_image_file(p)
        ]
    )


def get_class_directories(root: Path) -> List[Path]:
    class_dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            images = list_images(p)
            if images:
                class_dirs.append(p)
    return class_dirs


def copy_original_dataset(src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for class_dir in get_class_directories(src_root):
        relative = class_dir.relative_to(src_root)
        target_class_dir = dst_root / relative
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for image_path in list_images(class_dir):
            target = target_class_dir / image_path.name
            if not target.exists():
                Image.open(image_path).save(target)


def class_distribution(root: Path) -> Dict[Path, int]:
    distribution: Dict[Path, int] = {}
    for class_dir in get_class_directories(root):
        distribution[class_dir] = len(list_images(class_dir))
    return distribution


def choose_source_image(images: List[Path], index: int) -> Path:
    return images[index % len(images)]


def balance_directory(
    src_root: Path,
    output_root: Path,
    seed: int,
    show_preview: bool,
) -> None:
    rng = random.Random(seed)
    copy_original_dataset(src_root, output_root)
    out_classes = get_class_directories(output_root)
    if not out_classes:
        raise ValueError(f"No image class found in: {src_root}")

    counts = {class_dir: len(list_images(class_dir))
              for class_dir in out_classes}
    target_count = max(counts.values())

    for class_dir in out_classes:
        current_images = list_images(class_dir)

        need = target_count - len(current_images)
        if need <= 0:
            continue

        source_pool = [
            p for p in current_images if is_augmented_filename(p) == False]
        if not source_pool:
            source_pool = current_images

        for i in range(need):
            source = choose_source_image(source_pool, i)
            aug = AUGMENTATIONS[i % len(AUGMENTATIONS)]
            image = ensure_rgb(Image.open(source))
            transformed = aug.func(image, rng)
            source_base_name = base_stem_without_augmentation(source)
            out_name = (
                f"{source_base_name}_{aug.name}_"
                f"{i + 1:04d}{source.suffix}"
            )
            out_path = class_dir / out_name
            collision_index = i + 1
            while out_path.exists():
                collision_index += 1
                out_name = (
                    f"{source_base_name}_{aug.name}_"
                    f"{collision_index:04d}{source.suffix}"
                )
                out_path = class_dir / out_name
            transformed.save(out_path)
        print(
            f"[BALANCE] {class_dir.name}: "
            f"{len(list_images(class_dir))}/{target_count}"
        )

    if show_preview:
        for class_dir in out_classes:
            images = list_images(class_dir)
            if not images:
                continue
            sample = ensure_rgb(Image.open(images[0]))
            preview_map = {aug.name: aug.func(
                sample, rng) for aug in AUGMENTATIONS}
            display_augmentations(sample, preview_map, class_dir.name)

    print(f"\nBalanced dataset created in: {output_root}")
    final_counts = {
        class_dir.name: len(list_images(class_dir))
        for class_dir in out_classes
    }
    for class_name, count in sorted(final_counts.items()):
        print(f"- {class_name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 2: Data augmentation and class balancing."
    )
    parser.add_argument(
        "path",
        help=(
            "Path to one image (single-image mode) or a class directory "
            "(balancing mode)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output directory. Default: same directory (image) or "
            "./augmented_directory/<dataset_name>."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the matplotlib preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.path).expanduser().resolve()
    show = not args.no_show
    if input_path.is_file():
        output_dir = Path(args.output).expanduser(
        ).resolve() if args.output else None
        created = generate_six_for_image(
            image_path=input_path,
            output_dir=output_dir,
            seed=args.seed,
            show=show,
        )
        print("Generated images:")
        for aug in AUGMENTATIONS:
            print(f"- {aug.name}: {created[aug.name]}")
        return
    if input_path.is_dir():
        if args.output:
            out_root = Path(args.output).expanduser().resolve()
        else:
            out_root = Path.cwd() / "augmented_directory" / input_path.name
        balance_directory(
            src_root=input_path,
            output_root=out_root,
            seed=args.seed,
            show_preview=False,
        )
        return
    raise FileNotFoundError(f"Path not found: {input_path}")


if __name__ == "__main__":
    main()
