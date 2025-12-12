from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class edge_sample:
    rel_x: int = 0
    rel_y: int = 0
    blue: int = 0
    green: int = 0
    red: int = 0


@dataclass
class piece_fragment:
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    border_trace: List[edge_sample] = field(default_factory=list)
    bitmap: Optional[np.ndarray] = None
    spin_hint: float = 0.0
    target_box: Tuple[int, int] = (0, 0)
    grid_slot: Optional[Tuple[int, int]] = None
    start_anchor: Tuple[float, float] = (0.0, 0.0)
    final_anchor: Tuple[float, float] = (0.0, 0.0)
    start_spin: float = 0.0
    final_spin: float = 0.0
    mask: Optional[np.ndarray] = None

    @property
    def col(self) -> Optional[int]:
        if self.grid_slot is None:
            return None
        return self.grid_slot[0]

    @property
    def row(self) -> Optional[int]:
        if self.grid_slot is None:
            return None
        return self.grid_slot[1]

    def attach_grid_slot(self, location: Tuple[int, int]) -> None:
        self.grid_slot = location

    def ensure_mask(self) -> None:
        if self.bitmap is None or self.mask is not None:
            return
        gray = cv2.cvtColor(self.bitmap, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        self.mask = thresholded
