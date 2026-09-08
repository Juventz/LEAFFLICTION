# predict.py
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from Transformation import read_rgb_image, compute_transformations, TransformResult


# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
DEFAULT_ARTIFACTS_DIR = Path("model_artifacts")


# ── Load artifacts ────────────────────────────────────────────────────────────
def load_artifacts(artifacts_dir: Path) -> tuple[tf.keras.Model, list[str]]:
    model_path = artifacts_dir / "best_model.keras"
    class_names_path = artifacts_dir / "class_names.txt"

    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        sys.exit(1)
    if not class_names_path.exists():
        print(f"Error: class_names.txt not found at {class_names_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(str(model_path))
    class_names = class_names_path.read_text().strip().splitlines()
    return model, class_names


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model: tf.keras.Model, class_names: list[str], image_path: Path) -> tuple[str, float]:
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img_batch = tf.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    predictions = model.predict(img_batch, verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][idx])
    return class_names[idx], confidence


# ── Display ───────────────────────────────────────────────────────────────────
def display(
    original: np.ndarray,
    transforms: list[TransformResult],
    predicted_class: str,
    confidence: float,
) -> None:
    panels = [("Original", original)] + [(t.name, t.image) for t in transforms]

    n_cols = 4
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(name)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(
        f"=== DL classification ===\nClass predicted: {predicted_class} ({confidence:.1%})",
        fontsize=14,
        fontweight="bold",
        color="royalblue",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig("prediction.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Prediction saved to prediction.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 4: predict leaf disease class from a trained model.",
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the leaf image to classify.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing best_model.keras and class_names.txt (default: {DEFAULT_ARTIFACTS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    image_path = args.image_path

    if not artifacts_dir.is_dir():
        print(f"Error: artifacts directory not found: {artifacts_dir}")
        sys.exit(1)
    if not image_path.is_file():
        print(f"Error: {image_path} is not a valid file")
        sys.exit(1)

    model, class_names = load_artifacts(artifacts_dir)
    predicted_class, confidence = predict(model, class_names, image_path)

    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.1%}")

    rgb_img = read_rgb_image(image_path)
    transforms = compute_transformations(rgb_img)
    display(rgb_img, transforms, predicted_class, confidence)


if __name__ == "__main__":
    main()