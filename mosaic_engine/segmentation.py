from typing import List, Tuple

import cv2
import numpy as np

from .models import edge_sample, piece_fragment


def _locate_shapes(frame: np.ndarray) -> List[np.ndarray]:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def _realign_quadrilateral(
    frame: np.ndarray, outline: np.ndarray, approx: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[int, int]], float, Tuple[float, float], Tuple[int, int]]:
    x, y, w, h = cv2.boundingRect(approx)
    rect = cv2.minAreaRect(approx)
    (center_x, center_y), (width, height), angle = rect
    width, height = int(width), int(height)
    angle = int(round(angle))

    aligned = frame[y : y + h, x : x + w].copy()
    spin = angle

    if angle not in (0, 90):
        corners = cv2.boxPoints(rect).astype(np.float32)
        sorted_by_y = sorted(corners.tolist(), key=lambda pt: pt[1])
        if sorted_by_y[2][0] < sorted_by_y[0][0]:
            spin += 90

        if 90 < spin < 180:
            destination = np.float32(
                [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]]
            )
        elif 0 < spin < 90:
            destination = np.float32(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            )
        else:
            destination = None

        if destination is not None:
            transform = cv2.getPerspectiveTransform(corners, destination)
            warped = cv2.warpPerspective(
                frame,
                transform,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            aligned = warped[1:-1, 1:-1]

    normalized_polygon = [
        (int(pt[0] - x), int(pt[1] - y)) for pt in approx.reshape(-1, 2)
    ]
    return aligned, normalized_polygon, spin, (center_x, center_y), (x, y)


def _trace_outline(frame: np.ndarray, contour: np.ndarray, origin: Tuple[int, int]) -> List[edge_sample]:
    origin_x, origin_y = origin
    samples: List[edge_sample] = []
    for point in contour:
        px, py = point[0]
        blue, green, red = frame[py, px]
        samples.append(edge_sample(px - origin_x, py - origin_y, int(blue), int(green), int(red)))
    return samples


def extract_fragments(frame: np.ndarray) -> List[piece_fragment]:
    """Isolate each puzzle piece from the source frame."""
    fragments: List[piece_fragment] = []
    contours = _locate_shapes(frame)

    for outline in contours:
        simplified = cv2.approxPolyDP(outline, 0.01 * cv2.arcLength(outline, True), True)
        if len(simplified) != 4:
            continue

        cropped, polygon, spin, center, origin = _realign_quadrilateral(frame, outline, simplified)
        edge_pixels = _trace_outline(frame, outline, origin)

        fragment = piece_fragment(
            polygon=polygon,
            border_trace=edge_pixels,
            bitmap=cropped,
            spin_hint=spin,
        )
        fragment.start_spin = spin - 90
        fragment.start_anchor = center
        fragment.ensure_mask()

        fragments.append(fragment)

    return fragments
