"""Piece extraction from scrambled puzzle images."""

from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np

from ..utils.piece import Piece


def extract_pieces(
    image: np.ndarray,
    expected_count: int,
    background_threshold: int = 10,
    allow_grid_fallback: bool = True
) -> List[Piece]:
    """Extract individual puzzle pieces from a scrambled image.
    
    Uses thresholding to isolate non-black pixels (pieces) from the 
    black background, then finds contours to identify each piece.
    
    Args:
        image: Input image in BGR format
        expected_count: Expected number of pieces (N)
        background_threshold: Pixel intensity threshold for background detection
        
    Returns:
        List of Piece objects extracted from the image
        
    Raises:
        ValueError: If detected piece count doesn't match expected_count
    """
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to create binary mask (non-black pixels = foreground)
    _, binary = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out noise (very small contours)
    min_area = 100  # Minimum area for a valid piece
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Validate piece count
    detected_count = len(valid_contours)
    if detected_count != expected_count:
        # Fallback: if we only found one large contour (likely a tiled image with no gaps),
        # optionally slice the canvas into a regular grid based on expected_count.
        if allow_grid_fallback and detected_count == 1:
            return _slice_into_grid(image, expected_count)
        raise ValueError(
            f"Piece count mismatch: expected {expected_count}, detected {detected_count}. "
            f"Check your image or adjust the --n parameter."
        )
    
    # Extract each piece
    pieces = []
    for idx, contour in enumerate(valid_contours):
        piece = _extract_single_piece(image, contour, idx)
        pieces.append(piece)
    
    # Sort pieces by their initial position (top-left to bottom-right)
    pieces.sort(key=lambda p: (p.initial_center[1], p.initial_center[0]))
    
    # Reassign IDs after sorting
    for idx, piece in enumerate(pieces):
        piece.id = idx
    
    return pieces


def _extract_single_piece(
    image: np.ndarray,
    contour: np.ndarray,
    piece_id: int
) -> Piece:
    """Extract a single piece from the image given its contour.
    
    Args:
        image: Source image in BGR format
        contour: Contour points defining the piece boundary
        piece_id: Unique identifier for the piece
        
    Returns:
        Piece object with extracted image and position
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate center
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Extract piece image with bounding box
    piece_image = image[y:y+h, x:x+w].copy()
    
    # Create mask for the piece (to handle non-rectangular pieces)
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - np.array([x, y])
    cv2.drawContours(mask, [shifted_contour], 0, 255, -1)
    
    # Apply mask to piece image (set background to black)
    piece_image_masked = cv2.bitwise_and(piece_image, piece_image, mask=mask)
    
    # Detect initial rotation (estimate based on contour analysis)
    initial_rotation = _estimate_rotation(contour)
    
    return Piece(
        id=piece_id,
        image=piece_image_masked,
        initial_center=(center_x, center_y),
        initial_rotation=initial_rotation
    )


def _estimate_rotation(contour: np.ndarray) -> float:
    """Estimate the rotation angle of a piece from its contour.
    
    For rectangular pieces, uses minimum area rectangle to estimate rotation.
    Returns angle normalized to 0, 90, 180, or 270 degrees.
    
    Args:
        contour: Contour points
        
    Returns:
        Estimated rotation angle (0, 90, 180, or 270)
    """
    # Get minimum area rectangle
    rect = cv2.minAreaRect(contour)
    angle = rect[2]  # Angle in range [-90, 0)
    
    # Normalize angle to 0, 90, 180, 270
    # Note: minAreaRect returns angle in [-90, 0), we need to handle this
    if angle < -45:
        angle = angle + 90
    
    # Snap to nearest 90-degree increment
    angle = round(angle / 90) * 90
    angle = angle % 360
    
    return float(angle)


def _slice_into_grid(image: np.ndarray, expected_count: int) -> List[Piece]:
    """Fallback segmentation that slices the canvas into a uniform grid.
    
    Useful when pieces touch (contours merge) or the input is already a tiled grid.
    """
    h, w = image.shape[:2]
    
    # Derive grid dims from expected_count (factor closest to square)
    rows = int(np.sqrt(expected_count))
    while expected_count % rows != 0 and rows > 1:
        rows -= 1
    cols = expected_count // rows
    
    piece_w = w // cols
    piece_h = h // rows
    
    pieces: List[Piece] = []
    pid = 0
    for r in range(rows):
        for c in range(cols):
            x1 = c * piece_w
            y1 = r * piece_h
            x2 = w if c == cols - 1 else (c + 1) * piece_w
            y2 = h if r == rows - 1 else (r + 1) * piece_h
            tile = image[y1:y2, x1:x2].copy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            pieces.append(Piece(
                id=pid,
                image=tile,
                initial_center=(center_x, center_y),
                initial_rotation=0.0
            ))
            pid += 1
    
    return pieces


def detect_piece_count(
    image: np.ndarray,
    background_threshold: int = 10,
    min_area: int = 100
) -> int:
    """Estimate the number of pieces in a scrambled image.
    
    Uses the same threshold/contour logic as extraction. If contours merge
    into a single blob (e.g., pre-tiled grid), falls back to detecting grid
    lines based on gradient peaks to infer rows/cols.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    detected = len(valid_contours)
    if detected > 1:
        return detected
    
    # Single blob: try to infer grid by detecting boundary lines
    rows, cols = _detect_grid_from_edges(gray)
    if rows * cols > 1:
        return rows * cols
    
    # Worst-case fallback
    return max(1, detected)


def _detect_grid_from_edges(gray: np.ndarray) -> Tuple[int, int]:
    """Detect grid rows/cols from edge responses when pieces touch."""
    # Compute mean absolute diff along axes
    col_diff = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean(axis=0)
    row_diff = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1)
    
    cols = _count_segments(col_diff)
    rows = _count_segments(row_diff)
    
    return rows, cols


def _count_segments(diff: np.ndarray, std_factor: float = 2.0, min_gap: int = 2) -> int:
    """Count segments separated by peaks in diff signal."""
    if diff.size == 0:
        return 1
    
    threshold = float(diff.mean() + std_factor * diff.std())
    indices = np.where(diff > threshold)[0]
    if len(indices) == 0:
        return 1
    
    # Group contiguous peaks
    groups = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx - prev > min_gap:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    
    # Number of segments = number of separators + 1
    return len(groups) + 1


def get_piece_size(image: np.ndarray, expected_count: int) -> Tuple[int, int]:
    """Estimate the size of individual pieces.
    
    Useful for validation and grid calculations.
    
    Args:
        image: Input image
        expected_count: Expected number of pieces
        
    Returns:
        Tuple of (piece_width, piece_height)
    """
    h, w = image.shape[:2]
    
    # Assume square grid for simplicity
    grid_size = int(np.sqrt(expected_count))
    if grid_size * grid_size != expected_count:
        # Non-square grid, estimate based on aspect ratio
        aspect = w / h
        rows = int(np.sqrt(expected_count / aspect))
        cols = int(expected_count / rows)
        return w // cols, h // rows
    
    return w // grid_size, h // grid_size
