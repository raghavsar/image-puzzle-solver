"""Piece extraction from scrambled puzzle images."""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from ..utils.piece import Piece


def extract_pieces(
    image: np.ndarray,
    expected_count: int,
    background_threshold: int = 10
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
