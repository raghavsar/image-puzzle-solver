"""Image I/O utilities for loading RGB and PNG files."""

import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image(filepath: Union[str, Path], width: int = 800, height: int = 800) -> np.ndarray:
    """Load an image from file (supports .png, .jpg, .rgb formats).
    
    Args:
        filepath: Path to the image file
        width: Expected width for .rgb files (default 800)
        height: Expected height for .rgb files (default 800)
        
    Returns:
        Image as numpy array in BGR format (OpenCV convention)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")
    
    ext = filepath.suffix.lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        # Use OpenCV for standard image formats
        image = cv2.imread(str(filepath))
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        return image
    
    elif ext == '.rgb':
        # Raw RGB format: read binary and reshape
        image = load_rgb_file(filepath, width, height)
        return image
    
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def load_rgb_file(filepath: Union[str, Path], width: int = 800, height: int = 800) -> np.ndarray:
    """Load a raw .rgb file.
    
    The .rgb format uses planar storage:
    - First all R values (width * height bytes)
    - Then all G values (width * height bytes)
    - Then all B values (width * height bytes)
    - Total size: width * height * 3 bytes
    
    Args:
        filepath: Path to the .rgb file
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Image as numpy array in BGR format (OpenCV convention)
        
    Raises:
        ValueError: If file size doesn't match expected dimensions
    """
    filepath = Path(filepath)
    
    expected_size = width * height * 3
    actual_size = filepath.stat().st_size
    
    if actual_size != expected_size:
        raise ValueError(
            f"RGB file size mismatch. Expected {expected_size} bytes "
            f"({width}x{height}x3), got {actual_size} bytes"
        )
    
    # Read raw bytes
    data = np.fromfile(str(filepath), dtype=np.uint8)
    
    # Reshape as planar (3 planes: R, G, B)
    planar = data.reshape(3, height, width)
    
    # Transpose to (height, width, 3) format
    image_rgb = np.transpose(planar, (1, 2, 0))
    
    # Convert RGB to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def save_image(image: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save an image to file.
    
    Args:
        image: Image as numpy array in BGR format
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filepath), image)
