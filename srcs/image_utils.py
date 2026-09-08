from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"})


def is_image_file(path: str | Path) -> bool:
    """Return True when the path has a supported image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS
