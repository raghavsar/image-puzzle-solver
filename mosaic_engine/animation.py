from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .models import piece_fragment

DEFAULT_CANVAS = (800, 800)


def _aggregate_extents(fragments: List[piece_fragment]) -> Tuple[List[float], List[float]]:
    x_values: List[float] = []
    y_values: List[float] = []
    for fragment in fragments:
        x_values.extend([fragment.start_anchor[0], fragment.final_anchor[0]])
        y_values.extend([fragment.start_anchor[1], fragment.final_anchor[1]])
    return x_values, y_values


def _scale_and_offset(fragments: List[piece_fragment], canvas: Tuple[int, int]) -> Tuple[float, float, float]:
    width, height = canvas
    xs, ys = _aggregate_extents(fragments)
    if not xs or not ys:
        return 1.0, 0.0, 0.0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin = 50
    span_x = max_x - min_x if max_x > min_x else 1
    span_y = max_y - min_y if max_y > min_y else 1

    scale_x = (width - 2 * margin) / span_x if span_x > 0 else 1.0
    scale_y = (height - 2 * margin) / span_y if span_y > 0 else 1.0
    scale = min(scale_x, scale_y, 1.0)

    offset_x = (width - (max_x + min_x) * scale) / 2
    offset_y = (height - (max_y + min_y) * scale) / 2

    return scale, offset_x, offset_y


def _rotate_fragment(bitmap: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = bitmap.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_theta = abs(matrix[0, 0])
    sin_theta = abs(matrix[0, 1])
    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)

    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]

    rotated_bitmap = cv2.warpAffine(bitmap, matrix, (new_width, new_height), borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, matrix, (new_width, new_height), borderValue=0)
    return rotated_bitmap, rotated_mask


def emit_animation(
    fragments: List[piece_fragment],
    source_frame: np.ndarray | None,
    frame_count: int = 30,
    output_filename: str = "puzzle_solution.mp4",
) -> None:
    """Interpolate piece movement from detected state to solved grid and emit an MP4."""
    if source_frame is not None and len(source_frame.shape) >= 2:
        canvas_height, canvas_width = source_frame.shape[:2]
    else:
        canvas_height, canvas_width = DEFAULT_CANVAS

    scale, offset_x, offset_y = _scale_and_offset(fragments, (canvas_width, canvas_height))

    path = Path(output_filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    fps = 15
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_width, canvas_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_filename}")

    for frame_idx in range(frame_count):
        progress = 0 if frame_count == 1 else frame_idx / (frame_count - 1)
        frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for fragment in fragments:
            if fragment.bitmap is None:
                continue

            # Resize to the same target cell used for the solved grid to avoid gaps.
            if fragment.target_box != (0, 0):
                target_w = int(fragment.target_box[0])
                target_h = int(fragment.target_box[1])
            else:
                target_w = int(fragment.bitmap.shape[1])
                target_h = int(fragment.bitmap.shape[0])

            if (fragment.bitmap.shape[1], fragment.bitmap.shape[0]) != (target_w, target_h):
                tile_bitmap = cv2.resize(fragment.bitmap, (target_w, target_h))
            else:
                tile_bitmap = fragment.bitmap

            tile_mask = fragment.mask
            if tile_mask is None:
                fragment.ensure_mask()
                tile_mask = fragment.mask
            if tile_mask is not None and tile_mask.shape[:2] != (target_h, target_w):
                tile_mask = cv2.resize(tile_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            start_x, start_y = fragment.start_anchor
            end_x, end_y = fragment.final_anchor
            anchor_x = int((start_x * (1 - progress) + end_x * progress) * scale + offset_x)
            anchor_y = int((start_y * (1 - progress) + end_y * progress) * scale + offset_y)

            start_spin = fragment.start_spin
            end_spin = fragment.final_spin
            angle = start_spin * (1 - progress) + end_spin * progress

            if tile_mask is None:
                continue

            rotated_bitmap, rotated_mask = _rotate_fragment(tile_bitmap, tile_mask, angle)
            height, width = rotated_bitmap.shape[:2]

            x1 = int(anchor_x - width / 2)
            y1 = int(anchor_y - height / 2)
            x2 = x1 + width
            y2 = y1 + height

            target_x1 = max(x1, 0)
            target_y1 = max(y1, 0)
            target_x2 = min(x2, canvas_width)
            target_y2 = min(y2, canvas_height)

            if target_x1 >= target_x2 or target_y1 >= target_y2:
                continue

            src_x1 = target_x1 - x1
            src_y1 = target_y1 - y1
            src_x2 = src_x1 + (target_x2 - target_x1)
            src_y2 = src_y1 + (target_y2 - target_y1)

            crop_bitmap = rotated_bitmap[src_y1:src_y2, src_x1:src_x2]
            crop_mask = rotated_mask[src_y1:src_y2, src_x1:src_x2] / 255.0
            crop_mask = crop_mask[:, :, None]

            frame[target_y1:target_y2, target_x1:target_x2] = (
                crop_bitmap * crop_mask
                + frame[target_y1:target_y2, target_x1:target_x2] * (1 - crop_mask)
            ).astype(np.uint8)

        writer.write(frame)

    writer.release()
