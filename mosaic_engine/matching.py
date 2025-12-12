from typing import Dict, List, Tuple

import cv2
import numpy as np

from .models import piece_fragment


OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def edge_vectors(fragment: piece_fragment) -> Dict[str, np.ndarray]:
    image = fragment.bitmap
    height, width = image.shape[:2]
    return {
        "north": image[0, :, :].astype(np.float32),
        "south": image[height - 1, :, :].astype(np.float32),
        "west": image[:, 0, :].astype(np.float32),
        "east": image[:, width - 1, :].astype(np.float32),
    }


def _complements(label_a: str, label_b: str) -> bool:
    return OPPOSITE.get(label_a) == label_b


def score_edge_alignment(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    if edge_a.shape != edge_b.shape:
        edge_b = cv2.resize(edge_b, (edge_a.shape[1], edge_a.shape[0]))
    delta = np.subtract(edge_a, edge_b, dtype=np.float32)
    mse = np.mean(delta * delta)
    return -mse


def compute_pairings(fragments: List[piece_fragment]) -> Dict[int, Dict[int, Tuple[float, Tuple[str, str] | None]]]:
    fingerprints = [edge_vectors(fragment) for fragment in fragments]
    pairings: Dict[int, Dict[int, Tuple[float, Tuple[str, str] | None]]] = {}

    for idx in range(len(fragments)):
        pairings[idx] = {}
        for candidate in range(len(fragments)):
            if idx == candidate:
                continue
            top_score = -np.inf
            winning_dirs: Tuple[str, str] | None = None
            for label_a, edge_a in fingerprints[idx].items():
                for label_b, edge_b in fingerprints[candidate].items():
                    if not _complements(label_a, label_b):
                        continue
                    score = score_edge_alignment(edge_a, edge_b)
                    if score > top_score:
                        top_score = score
                        winning_dirs = (label_a, label_b)
            pairings[idx][candidate] = (top_score, winning_dirs)

    return pairings
