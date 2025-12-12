from typing import List, Tuple

import numpy as np

from .matching import compute_pairings
from .models import piece_fragment
from .placement import arrange_on_grid
from .render import compose_grid_image


def solve_and_render(fragments: List[piece_fragment], source_frame: np.ndarray) -> Tuple[List[piece_fragment], np.ndarray]:
    pairings = compute_pairings(fragments)
    arranged, _, _ = arrange_on_grid(fragments, pairings)
    canvas = compose_grid_image(arranged)
    return arranged, canvas
