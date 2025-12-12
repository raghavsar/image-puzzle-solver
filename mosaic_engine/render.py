from typing import List, Tuple

import cv2
import numpy as np

from .models import piece_fragment


def _grid_span(fragments: List[piece_fragment]) -> Tuple[int, int]:
    rows = [fragment.row for fragment in fragments if fragment.row is not None]
    cols = [fragment.col for fragment in fragments if fragment.col is not None]
    max_row = max(rows) if rows else 0
    max_col = max(cols) if cols else 0
    return max_row + 1, max_col + 1


def compose_grid_image(fragments: List[piece_fragment]) -> np.ndarray:
    """Render the assembled puzzle on a normalized canvas."""
    if not fragments:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    # Prefer per-fragment cell sizes; fall back to a universal max if missing.
    target_heights = [
        fragment.target_box[1]
        for fragment in fragments
        if fragment.bitmap is not None and fragment.target_box != (0, 0)
    ]
    target_widths = [
        fragment.target_box[0]
        for fragment in fragments
        if fragment.bitmap is not None and fragment.target_box != (0, 0)
    ]

    if target_heights and target_widths:
        cell_height = max(target_heights)
        cell_width = max(target_widths)
    else:
        cell_height = max(fragment.bitmap.shape[0] for fragment in fragments if fragment.bitmap is not None)
        cell_width = max(fragment.bitmap.shape[1] for fragment in fragments if fragment.bitmap is not None)

    rows, cols = _grid_span(fragments)
    canvas = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)

    for fragment in fragments:
        if fragment.bitmap is None or fragment.grid_slot is None:
            continue

        col, row = fragment.grid_slot
        y_start = row * cell_height
        y_end = y_start + cell_height
        x_start = col * cell_width
        x_end = x_start + cell_width

        target_size = fragment.target_box if fragment.target_box != (0, 0) else (cell_width, cell_height)
        resized = cv2.resize(fragment.bitmap, target_size)
        resized_mask = fragment.mask
        if resized_mask is not None and resized_mask.shape[:2] != target_size[::-1]:
            resized_mask = cv2.resize(resized_mask, target_size, interpolation=cv2.INTER_NEAREST)

        if resized_mask is not None:
            mask = resized_mask.astype(bool)
            cell = canvas[y_start:y_end, x_start:x_end]
            cell[mask] = resized[mask]
        else:
            canvas[y_start:y_end, x_start:x_end] = resized

    return canvas
