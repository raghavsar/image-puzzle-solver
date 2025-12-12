from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DEFAULT_RGB_DIMENSION = 800


def _deduce_rgb_dimension(raw: np.ndarray) -> int:
    pixel_count = raw.size // 3
    side = int(np.sqrt(pixel_count))
    if side * side * 3 != raw.size:
        return DEFAULT_RGB_DIMENSION
    return side


def load_frame(path: Path | str, fallback_shape: Tuple[int, int] = (DEFAULT_RGB_DIMENSION, DEFAULT_RGB_DIMENSION)) -> np.ndarray:
    """Read a puzzle source image from disk, accommodating both PNG and raw RGB dumps."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input not found: {file_path}")

    extension = file_path.suffix.lower()
    if extension == ".rgb":
        raw = np.fromfile(file_path, dtype=np.uint8)
        if raw.size == 0:
            raise ValueError(f"Unable to decode RGB file: {file_path}")
        side = _deduce_rgb_dimension(raw)
        try:
            buffer = raw.reshape(3, side, side)
        except ValueError as exc:
            raise ValueError(f"RGB payload has unexpected dimensions for {file_path}") from exc
        frame = np.transpose(buffer, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.imread(str(file_path))

    if frame is None:
        raise ValueError(f"Unsupported or corrupt image: {file_path}")

    return frame
