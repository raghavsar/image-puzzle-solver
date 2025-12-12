"""Piece extraction from scrambled puzzle images.

Supports both translation-only and rotation+translation puzzles.
For rotated pieces, uses perspective transform to extract axis-aligned tile images
while preserving the original rotation angle for animation.
"""

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
    Handles both rotated and non-rotated pieces.
    
    For rotation puzzles, specifically filters for quadrilateral shapes
    which are the actual puzzle pieces.
    
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
    
    # Find contours - use CHAIN_APPROX_NONE for rotation puzzles to get all points
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Calculate expected piece area based on image and piece count
    # Note: Actual pieces may be smaller due to gaps, so use very lenient bounds
    total_area = image.shape[0] * image.shape[1]
    expected_piece_area = total_area / expected_count
    # Use very lenient bounds - pieces can be 5% to 200% of the naive expected area
    # (gaps between pieces make actual pieces smaller)
    min_area = expected_piece_area * 0.05
    max_area = expected_piece_area * 2.0
    
    # Filter contours: keep only those that are roughly the right size
    # and approximate to quadrilaterals (4-sided shapes)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # For rotation puzzles, pieces should be quadrilaterals (4 sides)
        # But also accept slightly irregular shapes (3-6 sides)
        if 3 <= len(approx) <= 6:
            valid_contours.append(contour)
    
    # Validate piece count
    detected_count = len(valid_contours)
    if detected_count != expected_count:
        # Try with more relaxed filtering if we got too few
        if detected_count < expected_count:
            # Use original contour list with just basic area filtering
            basic_min_area = 100  # Just filter noise
            valid_contours = [c for c in contours if cv2.contourArea(c) > basic_min_area]
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
    
    For rotated pieces (quadrilaterals with non-axis-aligned edges), applies
    perspective transform to extract an axis-aligned tile image while 
    preserving the original rotation angle.
    
    Args:
        image: Source image in BGR format
        contour: Contour points defining the piece boundary
        piece_id: Unique identifier for the piece
        
    Returns:
        Piece object with extracted image, position, rotation, and mask
    """
    # Approximate contour to polygon
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    
    # Get minimum area bounding rectangle and bounding box
    x, y, w, h = cv2.boundingRect(approx)
    rect = cv2.minAreaRect(approx)
    (center_x, center_y), (rect_width, rect_height), angle = rect
    rect_width = int(rect_width)
    rect_height = int(rect_height)
    
    # Check if piece is actually rotated using the external repo's method:
    # If angle is 0 or 90 (within tolerance), the piece is axis-aligned
    is_axis_aligned = _is_axis_aligned(angle)
    is_rotated = sides == 4 and not is_axis_aligned
    
    if is_rotated:
        # Extract rotated tile using perspective transform
        piece_image, mask, initial_rotation = _extract_rotated_tile(
            image, approx, rect, angle
        )
    else:
        # Axis-aligned extraction - no rotation
        piece_image = image[y:y+h, x:x+w].copy()
        
        # Create mask for the piece
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - np.array([x, y])
        cv2.drawContours(mask, [shifted_contour], 0, 255, -1)
        
        # Apply mask to piece image (set background to black)
        piece_image = cv2.bitwise_and(piece_image, piece_image, mask=mask)
        
        # For axis-aligned pieces, initial rotation is 0
        initial_rotation = 0.0
    
    return Piece(
        id=piece_id,
        image=piece_image,
        initial_center=(center_x, center_y),
        initial_rotation=initial_rotation,
        mask=mask
    )


def _is_axis_aligned(angle: float, tolerance: float = 3.0) -> bool:
    """Check if angle is close to axis-aligned (0 or multiple of 90 degrees).
    
    minAreaRect returns angle in the range [-90, 0), where:
    - 0 means axis-aligned
    - -90 means axis-aligned (vertical)
    - Values between indicate rotation
    """
    # Normalize angle to positive
    angle_norm = angle % 90
    # Check if close to 0 or 90 (which means axis-aligned)
    return angle_norm < tolerance or angle_norm > (90 - tolerance)


def _extract_rotated_tile(
    image: np.ndarray,
    approx: np.ndarray,
    rect: Tuple,
    angle: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract a rotated tile using perspective transform.
    
    This is the key function for handling rotation puzzles - it "unrotates"
    the tile to get an axis-aligned image while recording the original rotation.
    Based on the approach from CSCI-576-Multimedia-System-Project.
    
    Args:
        image: Source image
        approx: Approximated contour points
        rect: Minimum area rectangle (center, size, angle) from minAreaRect
        angle: Rectangle angle from minAreaRect
        
    Returns:
        Tuple of (unrotated tile image, mask, initial rotation angle)
    """
    # Get the 4 corners of the rotated rectangle
    (center_x, center_y), (rect_width, rect_height), _ = rect
    width = int(rect_width)
    height = int(rect_height)
    
    box = cv2.boxPoints(rect)
    box = np.float32(box)
    
    # Sort corners by y coordinate to determine orientation
    coords = sorted(box, key=lambda p: p[1])
    
    # Determine which way the piece is tilted based on corner positions
    # If the lower-left corner is to the left of the upper-left corner, it's tilted one way
    if coords[2][0] < coords[0][0]:
        # Tilted - need to add 90 to get proper angle
        angle_adjusted = angle + 90
    else:
        angle_adjusted = angle
    
    # Round to int for cleaner angle values
    angle_adjusted = int(round(angle_adjusted))
    
    # Compute initial rotation (the angle the piece was rotated from upright)
    # We store this so we can animate back to 0
    initial_rotation = angle_adjusted - 90
    
    # Define destination points for perspective transform
    # This "unrotates" the piece to axis-aligned orientation
    if 90 < angle_adjusted < 180:
        # Tilted right
        transform = np.float32([
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1]
        ])
    elif 0 < angle_adjusted < 90:
        # Tilted left
        transform = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
    else:
        # Edge case - shouldn't reach here since we already checked for axis-aligned
        transform = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        initial_rotation = 0.0
    
    # Apply perspective transform to unrotate the tile
    rotation_matrix = cv2.getPerspectiveTransform(box, transform)
    tile_img = cv2.warpPerspective(
        image, rotation_matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    # Feather the border to remove jagged edges (remove 1 pixel border)
    if width > 4 and height > 4:
        tile_img = tile_img[1:-1, 1:-1]
    
    # Create mask for the unrotated tile
    mask = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    
    return tile_img, mask, float(initial_rotation)


def _estimate_rotation_from_rect(rect: Tuple) -> float:
    """Estimate rotation angle from minAreaRect result.
    
    Args:
        rect: Result from cv2.minAreaRect (center, size, angle)
        
    Returns:
        Rotation angle normalized for puzzle solving
    """
    (_, _), (width, height), angle = rect
    
    # minAreaRect returns angle in [-90, 0)
    # Adjust based on aspect ratio
    if height > width:
        angle = angle + 90
    
    # Snap to nearest 90-degree increment if close
    angle_mod = angle % 90
    if angle_mod < 5 or angle_mod > 85:
        angle = round(angle / 90) * 90
    
    return float(angle % 360)


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
