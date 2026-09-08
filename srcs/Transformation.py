#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

from image_utils import is_image_file


@dataclass(frozen=True)
class TransformResult:
    name: str
    image: np.ndarray


def read_rgb_image(path: Path) -> np.ndarray:
    img, _, _ = pcv.readimage(filename=str(path), mode="rgb")
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def choose_mask(candidates: List[np.ndarray]) -> np.ndarray:
    best_mask = candidates[0]
    best_score = -1.0
    for mask in candidates:
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if ratio < 0.01 or ratio > 0.95:
            score = -1.0
        else:
            score = 1.0 - abs(ratio - 0.35)
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 500,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned


def build_leaf_mask(rgb_img: np.ndarray) -> np.ndarray:
    saturation = pcv.rgb2gray_hsv(rgb_img=rgb_img, channel="s")
    saturation = pcv.gaussian_blur(img=saturation, ksize=(5, 5))
    light_mask = pcv.threshold.otsu(gray_img=saturation, object_type="light")
    dark_mask = pcv.threshold.otsu(gray_img=saturation, object_type="dark")
    masks = []
    for m in (light_mask, dark_mask):
        filled = pcv.fill_holes(bin_img=m)
        cleaned = remove_small_components(filled, min_size=500)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        masks.append(cleaned)
    mask = choose_mask(masks)
    if np.count_nonzero(mask) == 0:
        value = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="l")
        fallback = pcv.threshold.otsu(gray_img=value, object_type="dark")
        mask = remove_small_components(
            pcv.fill_holes(bin_img=fallback),
            min_size=300,
        )
    return mask.astype(np.uint8)


def roi_objects_overlay(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb_img.copy()
    if len(np.unique(mask)) != 2:
        return overlay
    pcv.roi.from_binary_image(img=rgb_img, bin_img=mask)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(main_contour)
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
    return overlay


def analyze_object(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    labeled_mask, n_labels = pcv.create_labels(mask=mask)
    return pcv.analyze.size(
        img=rgb_img,
        labeled_mask=labeled_mask,
        n_labels=n_labels,
    )


def draw_landmark_points(
    img: np.ndarray,
    landmark_groups: Tuple[object, object, object],
    color: Tuple[int, int, int],
) -> None:
    for points in landmark_groups:
        array = np.asarray(points)
        if array.dtype.kind not in "iuf" or array.size == 0:
            continue
        for point in array.reshape(-1, 2):
            cv2.circle(img, tuple(np.rint(point).astype(int)), 2, color, -1)


def pseudolandmarks_overlay(
    rgb_img: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    overlay = rgb_img.copy()
    x_landmarks = pcv.homology.x_axis_pseudolandmarks(img=rgb_img, mask=mask)
    y_landmarks = pcv.homology.y_axis_pseudolandmarks(img=rgb_img, mask=mask)
    draw_landmark_points(overlay, x_landmarks, (255, 0, 0))
    draw_landmark_points(overlay, y_landmarks, (0, 255, 255))
    return overlay


def color_histogram(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, histogram_data = pcv.visualize.histogram(
        img=rgb_img,
        mask=mask,
        bins=32,
        title="Color Histogram",
        hist_data=True,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"red": "red", "green": "green", "blue": "blue"}
    for channel, color in colors.items():
        values = histogram_data[histogram_data["color channel"] == channel]
        ax.plot(
            values["pixel intensity"],
            values["proportion of pixels (%)"],
            color=color,
            label=channel.capitalize(),
        )
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    histogram_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return histogram_img


def compute_transformations(rgb_img: np.ndarray) -> List[TransformResult]:
    gaussian_blur = pcv.gaussian_blur(img=rgb_img, ksize=(5, 5))
    mask = build_leaf_mask(rgb_img=rgb_img)
    roi_objects = roi_objects_overlay(rgb_img=rgb_img, mask=mask)
    analyzed = analyze_object(rgb_img=rgb_img, mask=mask)
    landmarks = pseudolandmarks_overlay(rgb_img=rgb_img, mask=mask)
    histogram = color_histogram(rgb_img=rgb_img, mask=mask)
    return [
        TransformResult("GaussianBlur", gaussian_blur),
        TransformResult("Mask", mask),
        TransformResult("RoiObjects", roi_objects),
        TransformResult("AnalyzeObject", analyzed),
        TransformResult("Pseudolandmarks", landmarks),
        TransformResult("ColorHistogram", histogram),
    ]


def save_transformations(
    src_image: Path,
    transforms: List[TransformResult],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}
    for t in transforms:
        out_path = output_dir / f"{src_image.stem}_{t.name}{src_image.suffix}"
        img = t.image
        if img.ndim == 2:
            to_write = img
        else:
            to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), to_write)
        saved[t.name] = out_path
    return saved


def display_transformations(
    original: np.ndarray, transforms: List[TransformResult], title: str
) -> None:
    n_images = len(transforms) + 1
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, transform in enumerate(transforms, start=1):
        axes[i].imshow(
            transform.image,
            cmap="gray" if transform.image.ndim == 2 else None,
        )
        axes[i].set_title(transform.name)
        axes[i].axis("off")
    for i in range(len(transforms) + 1, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"Image Transformations - {title}", fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def process_single_image(
    image_path: Path,
    save_dir: Path | None,
    show: bool,
) -> None:
    rgb = read_rgb_image(path=image_path)
    transforms = compute_transformations(rgb_img=rgb)
    if save_dir is not None:
        saved = save_transformations(
            src_image=image_path,
            transforms=transforms,
            output_dir=save_dir,
        )
        print("Saved transformations:")
        for t in transforms:
            print(f"- {t.name}: {saved[t.name]}")
    if show:
        display_transformations(
            original=rgb, transforms=transforms, title=image_path.name)


def iter_images_recursive(src_dir: Path) -> List[Path]:
    return sorted(
        [p for p in src_dir.rglob("*") if p.is_file() and is_image_file(p)]
    )


def process_directory(
    src_dir: Path,
    dst_dir: Path,
    show: bool,
) -> None:
    images = iter_images_recursive(src_dir=src_dir)
    if not images:
        raise ValueError(f"No images found in: {src_dir}")
    for idx, image_path in enumerate(images, start=1):
        relative_parent = image_path.parent.relative_to(src_dir)
        output_dir = dst_dir / relative_parent
        rgb = read_rgb_image(path=image_path)
        transforms = compute_transformations(rgb_img=rgb)
        save_transformations(src_image=image_path,
                             transforms=transforms,
                             output_dir=output_dir)
        print(f"[{idx}/{len(images)}] OK -> {image_path}")
        if show and idx == 1:
            display_transformations(
                original=rgb, transforms=transforms, title=image_path.name)
    print(f"\nTransformations saved in: {dst_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Part 3: leaf image transformations with PlantCV."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to one image (display mode).",
    )
    parser.add_argument(
        "-src",
        help="Source image directory (batch mode).",
    )
    parser.add_argument(
        "-dst",
        help=(
            "Destination directory used to save transformations in batch "
            "mode."
        ),
    )
    parser.add_argument(
        "-mask",
        action="store_true",
        help=(
            "Subject-compatible option; the mask pipeline is already "
            "included."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "In image mode, also save the transformations next to the image."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable matplotlib figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show = not args.no_show
    if args.src:
        src_dir = Path(args.src).expanduser().resolve()
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {src_dir}")
        if not args.dst:
            raise ValueError("Batch mode requires -dst.")
        dst_dir = Path(args.dst).expanduser().resolve()
        process_directory(
            src_dir=src_dir,
            dst_dir=dst_dir,
            show=show,
        )
        return
    if not args.image:
        raise ValueError("Provide an image or use -src/-dst.")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_image_file(image_path):
        raise ValueError(f"Unsupported image file: {image_path}")
    save_dir = image_path.parent if args.save else None
    process_single_image(
        image_path=image_path,
        save_dir=save_dir,
        show=show,
    )


if __name__ == "__main__":
    main()
