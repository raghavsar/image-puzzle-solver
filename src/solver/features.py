"""Feature extraction utilities for puzzle piece edge matching.

This module implements histogram-based feature extraction for comparing
puzzle piece edges, including:
- Color histograms (per-channel RGB)
- Edge/gradient histograms using Sobel operators
- Combined feature descriptors

These features provide more robust matching than raw pixel comparison.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


# Default histogram parameters
DEFAULT_COLOR_BINS = 32
DEFAULT_GRADIENT_BINS = 16
GRADIENT_MAGNITUDE_THRESHOLD = 10


def compute_color_histogram(
    edge_pixels: np.ndarray,
    bins: int = DEFAULT_COLOR_BINS,
    normalize: bool = True
) -> np.ndarray:
    """Compute color histogram for edge pixels.
    
    Computes per-channel histograms for B, G, R channels and concatenates them
    into a single feature vector.
    
    Args:
        edge_pixels: Edge pixel array of shape (N, 3) in BGR format
        bins: Number of histogram bins per channel
        normalize: Whether to L1-normalize each histogram
        
    Returns:
        Concatenated histogram feature vector of shape (bins * 3,)
    """
    if edge_pixels.ndim == 1:
        # Reshape flattened array to (N, 3)
        edge_pixels = edge_pixels.reshape(-1, 3)
    
    histograms = []
    for channel in range(3):  # B, G, R
        hist, _ = np.histogram(
            edge_pixels[:, channel],
            bins=bins,
            range=(0, 256)
        )
        hist = hist.astype(np.float64)
        
        if normalize and hist.sum() > 0:
            hist = hist / hist.sum()
        
        histograms.append(hist)
    
    return np.concatenate(histograms)


def compute_gradient_histogram(
    image: np.ndarray,
    side: str,
    bins: int = DEFAULT_GRADIENT_BINS,
    edge_width: int = 3,
    normalize: bool = True
) -> np.ndarray:
    """Compute gradient/edge histogram for a piece edge.
    
    Uses Sobel operators to compute gradients along the edge region,
    then creates a histogram of gradient magnitudes and orientations.
    
    Args:
        image: Piece image in BGR format
        side: Which edge to analyze ('top', 'bottom', 'left', 'right')
        bins: Number of orientation bins
        edge_width: Width of edge region to analyze (in pixels)
        normalize: Whether to L1-normalize the histogram
        
    Returns:
        Gradient histogram feature vector of shape (bins,)
    """
    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    h, w = gray.shape
    
    # Extract edge region
    if side == 'top':
        region = gray[:edge_width, :]
    elif side == 'bottom':
        region = gray[max(0, h - edge_width):, :]
    elif side == 'left':
        region = gray[:, :edge_width]
    elif side == 'right':
        region = gray[:, max(0, w - edge_width):]
    else:
        raise ValueError(f"Invalid side: {side}")
    
    # Compute Sobel gradients
    grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)  # Range: [-pi, pi]
    
    # Normalize angle to [0, 2*pi) for binning
    angle = (angle + np.pi) % (2 * np.pi)
    
    # Filter by magnitude threshold to ignore flat regions
    valid_mask = magnitude > GRADIENT_MAGNITUDE_THRESHOLD
    valid_angles = angle[valid_mask]
    valid_magnitudes = magnitude[valid_mask]
    
    if len(valid_angles) == 0:
        # No significant gradients found
        return np.zeros(bins, dtype=np.float64)
    
    # Create weighted histogram (weighted by gradient magnitude)
    hist, _ = np.histogram(
        valid_angles,
        bins=bins,
        range=(0, 2 * np.pi),
        weights=valid_magnitudes
    )
    hist = hist.astype(np.float64)
    
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    
    return hist


def compute_texture_features(
    image: np.ndarray,
    side: str,
    edge_width: int = 5
) -> np.ndarray:
    """Compute texture features for a piece edge using local statistics.
    
    Computes statistical measures (mean, std, gradient) along the edge
    to capture texture patterns.
    
    Args:
        image: Piece image in BGR format
        side: Which edge to analyze
        edge_width: Width of edge region to analyze
        
    Returns:
        Texture feature vector of shape (9,) - [mean_BGR, std_BGR, grad_mean_BGR]
    """
    h, w = image.shape[:2]
    
    # Extract edge region
    if side == 'top':
        region = image[:edge_width, :, :]
    elif side == 'bottom':
        region = image[max(0, h - edge_width):, :, :]
    elif side == 'left':
        region = image[:, :edge_width, :]
    elif side == 'right':
        region = image[:, max(0, w - edge_width):, :]
    else:
        raise ValueError(f"Invalid side: {side}")
    
    region = region.astype(np.float64)
    
    # Compute statistics per channel
    means = np.mean(region, axis=(0, 1))  # (3,)
    stds = np.std(region, axis=(0, 1))    # (3,)
    
    # Compute gradient along the perpendicular direction
    if side in ['top', 'bottom']:
        grad = np.diff(region, axis=0)
    else:
        grad = np.diff(region, axis=1)
    
    grad_means = np.mean(np.abs(grad), axis=(0, 1)) if grad.size > 0 else np.zeros(3)
    
    return np.concatenate([means, stds, grad_means])


def extract_edge_features(
    image: np.ndarray,
    side: str,
    rotation: float = 0.0,
    include_color_hist: bool = True,
    include_gradient_hist: bool = True,
    include_texture: bool = True,
    color_bins: int = DEFAULT_COLOR_BINS,
    gradient_bins: int = DEFAULT_GRADIENT_BINS
) -> np.ndarray:
    """Extract comprehensive feature vector for a piece edge.
    
    Combines multiple feature types into a single descriptor:
    - Color histogram (32 * 3 = 96 dims by default)
    - Gradient histogram (16 dims by default)
    - Texture features (9 dims)
    
    Args:
        image: Piece image in BGR format
        side: Which edge to analyze
        rotation: Rotation angle to apply before feature extraction
        include_color_hist: Whether to include color histogram
        include_gradient_hist: Whether to include gradient histogram
        include_texture: Whether to include texture features
        color_bins: Number of bins for color histogram
        gradient_bins: Number of bins for gradient histogram
        
    Returns:
        Combined feature vector
    """
    # Apply rotation if needed
    if rotation != 0:
        image = _rotate_image(image, rotation)
    
    features = []
    
    # Extract edge pixels for color histogram
    h, w = image.shape[:2]
    if side == 'top':
        edge_pixels = image[0, :, :]
    elif side == 'bottom':
        edge_pixels = image[h - 1, :, :]
    elif side == 'left':
        edge_pixels = image[:, 0, :]
    elif side == 'right':
        edge_pixels = image[:, w - 1, :]
    else:
        raise ValueError(f"Invalid side: {side}")
    
    if include_color_hist:
        color_hist = compute_color_histogram(edge_pixels, bins=color_bins)
        features.append(color_hist)
    
    if include_gradient_hist:
        grad_hist = compute_gradient_histogram(image, side, bins=gradient_bins)
        features.append(grad_hist)
    
    if include_texture:
        texture = compute_texture_features(image, side)
        features.append(texture)
    
    return np.concatenate(features) if features else np.array([])


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        
    Returns:
        Rotated image
    """
    if angle == 0:
        return image
    
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    
    return rotated


def histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram intersection similarity.
    
    Higher values indicate more similar histograms.
    Range: [0, 1] for normalized histograms.
    
    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
        
    Returns:
        Intersection similarity score
    """
    return float(np.sum(np.minimum(hist1, hist2)))


def chi_squared_distance(hist1: np.ndarray, hist2: np.ndarray, eps: float = 1e-10) -> float:
    """Compute chi-squared distance between histograms.
    
    Lower values indicate more similar histograms.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        eps: Small constant to avoid division by zero
        
    Returns:
        Chi-squared distance
    """
    return float(np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps)))


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between histograms.
    
    Lower values indicate more similar histograms.
    Range: [0, inf) for normalized histograms.
    
    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
        
    Returns:
        Bhattacharyya distance
    """
    bc = np.sum(np.sqrt(hist1 * hist2))  # Bhattacharyya coefficient
    return float(-np.log(bc + 1e-10))


def compute_feature_distance(
    features1: np.ndarray,
    features2: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """Compute distance between two feature vectors.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        metric: Distance metric ('euclidean', 'cosine', 'chi_squared')
        
    Returns:
        Distance value (lower = more similar)
    """
    if len(features1) != len(features2):
        raise ValueError(
            f"Feature vectors have different lengths: {len(features1)} vs {len(features2)}"
        )
    
    if metric == 'euclidean':
        return float(np.linalg.norm(features1 - features2))
    
    elif metric == 'cosine':
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(1.0 - similarity)  # Convert to distance
    
    elif metric == 'chi_squared':
        return chi_squared_distance(features1, features2)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
