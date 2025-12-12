from typing import Dict, List, Tuple

import numpy as np

from .models import piece_fragment

OFFSETS = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}


def _pick_seed(pairings: Dict[int, Dict[int, Tuple[float, Tuple[str, str] | None]]]) -> Tuple[int, int, Tuple[str, str] | None]:
    best_score = -np.inf
    champion: Tuple[int, int, Tuple[str, str] | None] = (0, 0, None)
    for left_idx, neighbors in pairings.items():
        for right_idx, candidate in neighbors.items():
            if left_idx == right_idx:
                continue
            score, dirs = candidate
            if score > best_score:
                best_score = score
                champion = (left_idx, right_idx, dirs)
    return champion


def _normalize_slots(assignments: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    min_x = min(x for x, _ in assignments.values())
    min_y = min(y for _, y in assignments.values())
    return {idx: (x - min_x, y - min_y) for idx, (x, y) in assignments.items()}


def _cell_metrics(
    fragments: List[piece_fragment], slots: Dict[int, Tuple[int, int]]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    widths: Dict[int, int] = {}
    heights: Dict[int, int] = {}
    for idx, (col, row) in slots.items():
        image = fragments[idx].bitmap
        h, w = image.shape[:2]
        widths[col] = max(widths.get(col, 0), w)
        heights[row] = max(heights.get(row, 0), h)
    return widths, heights


def arrange_on_grid(
    fragments: List[piece_fragment],
    pairings: Dict[int, Dict[int, Tuple[float, Tuple[str, str] | None]]],
) -> Tuple[List[piece_fragment], int, int]:
    total = len(fragments)
    remaining = set(range(total))
    placements: Dict[int, Tuple[int, int]] = {}

    seed_a, seed_b, directions = _pick_seed(pairings)
    placements[seed_a] = (0, 0)
    if directions is not None:
        offset = OFFSETS.get(directions[0], (1, 0))
    else:
        offset = (1, 0)
    placements[seed_b] = (offset[0], offset[1])
    remaining.discard(seed_a)
    remaining.discard(seed_b)

    while remaining:
        chosen: int | None = None
        slot: Tuple[int, int] | None = None
        rating = -np.inf

        for candidate in remaining:
            for placed_idx, placed_slot in placements.items():
                score, dirs = pairings[placed_idx].get(candidate, (-np.inf, None))
                if dirs is None:
                    continue
                offset = OFFSETS.get(dirs[0])
                if offset is None:
                    continue
                destination = (placed_slot[0] + offset[0], placed_slot[1] + offset[1])
                if destination in placements.values():
                    continue
                if score > rating:
                    rating = score
                    chosen = candidate
                    slot = destination

        if chosen is None or slot is None:
            chosen = remaining.pop()
            fallback_slot = next(iter(placements.values()))
            slot = (fallback_slot[0] + 1, fallback_slot[1])
        else:
            remaining.remove(chosen)

        placements[chosen] = slot

    normalized = _normalize_slots(placements)

    for idx, slot in normalized.items():
        fragments[idx].attach_grid_slot(slot)

    max_col = max(col for col, _ in normalized.values())
    max_row = max(row for _, row in normalized.values())

    widths, heights = _cell_metrics(fragments, normalized)
    margin = 20

    col_offsets: Dict[int, float] = {}
    current = margin
    for col in range(0, max_col + 1):
        col_offsets[col] = current + widths.get(col, 0) / 2
        current += widths.get(col, 0)

    row_offsets: Dict[int, float] = {}
    current = margin
    for row in range(0, max_row + 1):
        row_offsets[row] = current + heights.get(row, 0) / 2
        current += heights.get(row, 0)

    for idx, (col, row) in normalized.items():
        # enforce integer target sizes for resizing downstream
        cell_w = int(widths.get(col, fragments[idx].bitmap.shape[1]))
        cell_h = int(heights.get(row, fragments[idx].bitmap.shape[0]))
        fragments[idx].target_box = (cell_w, cell_h)
        fragments[idx].final_anchor = (col_offsets[col], row_offsets[row])
        fragments[idx].final_spin = 0.0

    return fragments, max_row + 1, max_col + 1
