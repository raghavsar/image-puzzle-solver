"""Edge matching utilities for comparing puzzle piece edges.

Supports multiple matching methods:
- SSD (Sum of Squared Differences) on raw pixels
- Histogram-based matching using color and gradient histograms
- Combined feature matching with configurable weights
"""

from typing import Tuple, List, Dict, Optional
from enum import Enum
import numpy as np

from ..utils.piece import Piece
from .features import (
    extract_edge_features,
    compute_feature_distance,
    compute_color_histogram,
    compute_gradient_histogram,
    chi_squared_distance,
    histogram_intersection
)


class MatchingMethod(Enum):
    """Edge matching method selection."""
    SSD = "ssd"                    # Raw pixel SSD (original)
    HISTOGRAM = "histogram"        # Color + gradient histograms
    COMBINED = "combined"          # Weighted combination of SSD and histogram


# Valid rotation angles
ROTATIONS = [0, 90, 180, 270]

# Edge pairs that should match (piece_a_side, piece_b_side)
EDGE_PAIRS = {
    'right': 'left',
    'left': 'right',
    'bottom': 'top',
    'top': 'bottom'
}

# Default matching configuration
DEFAULT_METHOD = MatchingMethod.HISTOGRAM
SSD_WEIGHT = 0.3          # Weight for SSD in combined mode
HISTOGRAM_WEIGHT = 0.7    # Weight for histogram in combined mode


def compute_edge_cost(
    piece_a: Piece,
    piece_b: Piece,
    side_a: str,
    rotation_a: float,
    rotation_b: float,
    method: MatchingMethod = DEFAULT_METHOD
) -> float:
    """Compute the matching cost between two edges.
    
    Supports multiple matching methods:
    - SSD: Raw pixel sum of squared differences
    - HISTOGRAM: Color and gradient histogram comparison
    - COMBINED: Weighted combination of both
    
    Args:
        piece_a: First puzzle piece
        piece_b: Second puzzle piece
        side_a: Which side of piece_a to compare ('right', 'bottom')
        rotation_a: Rotation to apply to piece_a (degrees)
        rotation_b: Rotation to apply to piece_b (degrees)
        method: Matching method to use
        
    Returns:
        Matching cost (lower = better match)
    """
    # Get the corresponding side of piece_b
    side_b = EDGE_PAIRS.get(side_a)
    if side_b is None:
        raise ValueError(f"Invalid side: {side_a}")
    
    if method == MatchingMethod.SSD:
        return _compute_ssd_cost(piece_a, piece_b, side_a, side_b, rotation_a, rotation_b)
    elif method == MatchingMethod.HISTOGRAM:
        return _compute_histogram_cost(piece_a, piece_b, side_a, side_b, rotation_a, rotation_b)
    elif method == MatchingMethod.COMBINED:
        ssd_cost = _compute_ssd_cost(piece_a, piece_b, side_a, side_b, rotation_a, rotation_b)
        hist_cost = _compute_histogram_cost(piece_a, piece_b, side_a, side_b, rotation_a, rotation_b)
        # Normalize and combine (SSD tends to be much larger, so we scale it)
        return SSD_WEIGHT * (ssd_cost / 1e6) + HISTOGRAM_WEIGHT * hist_cost
    else:
        raise ValueError(f"Unknown matching method: {method}")


def _compute_ssd_cost(
    piece_a: Piece,
    piece_b: Piece,
    side_a: str,
    side_b: str,
    rotation_a: float,
    rotation_b: float
) -> float:
    """Compute SSD (Sum of Squared Differences) between two edges.
    
    Original matching method using raw pixel comparison.
    """
    # Extract edges with applied rotations
    edge_a = piece_a.get_edge(side_a, rotation_a)
    edge_b = piece_b.get_edge(side_b, rotation_b)
    
    # Ensure edges are the same length (resize if necessary)
    if len(edge_a) != len(edge_b):
        # Resize to match the smaller edge
        min_len = min(len(edge_a), len(edge_b))
        edge_a = _resize_edge(edge_a, min_len)
        edge_b = _resize_edge(edge_b, min_len)
    
    # Compute SSD
    diff = edge_a - edge_b
    ssd = np.sum(diff ** 2)
    
    return float(ssd)


def _resize_edge(edge: np.ndarray, target_len: int) -> np.ndarray:
    """Resize an edge array to target length using interpolation.
    
    Args:
        edge: Original edge array
        target_len: Target length
        
    Returns:
        Resized edge array
    """
    if len(edge) == target_len:
        return edge
    
    indices = np.linspace(0, len(edge) - 1, target_len)
    return np.interp(indices, np.arange(len(edge)), edge)


def _compute_histogram_cost(
    piece_a: Piece,
    piece_b: Piece,
    side_a: str,
    side_b: str,
    rotation_a: float,
    rotation_b: float
) -> float:
    """Compute histogram-based matching cost between two edges.
    
    Uses color histograms and gradient histograms for more robust matching
    than raw pixel comparison.
    
    Args:
        piece_a: First puzzle piece
        piece_b: Second puzzle piece
        side_a: Which side of piece_a to compare
        side_b: Which side of piece_b to compare
        rotation_a: Rotation to apply to piece_a (degrees)
        rotation_b: Rotation to apply to piece_b (degrees)
        
    Returns:
        Histogram distance (lower = better match)
    """
    # Extract features for both edges
    features_a = extract_edge_features(
        piece_a.image, side_a, rotation_a,
        include_color_hist=True,
        include_gradient_hist=True,
        include_texture=True
    )
    features_b = extract_edge_features(
        piece_b.image, side_b, rotation_b,
        include_color_hist=True,
        include_gradient_hist=True,
        include_texture=True
    )
    
    # Compute distance using chi-squared (good for histograms)
    return compute_feature_distance(features_a, features_b, metric='chi_squared')


def find_best_match(
    piece_a: Piece,
    piece_b: Piece,
    side: str = 'right',
    method: MatchingMethod = DEFAULT_METHOD
) -> Tuple[float, float, float]:
    """Find the best rotation combination for matching two pieces.
    
    Tests all 16 rotation combinations (4 rotations × 4 rotations) and
    returns the one with minimum matching cost.
    
    Args:
        piece_a: First puzzle piece
        piece_b: Second puzzle piece
        side: Which side of piece_a to match ('right' or 'bottom')
        method: Matching method to use
        
    Returns:
        Tuple of (min_cost, optimal_rotation_a, optimal_rotation_b)
    """
    min_cost = float('inf')
    best_rotation_a = 0.0
    best_rotation_b = 0.0
    
    for rot_a in ROTATIONS:
        for rot_b in ROTATIONS:
            cost = compute_edge_cost(piece_a, piece_b, side, rot_a, rot_b, method)
            if cost < min_cost:
                min_cost = cost
                best_rotation_a = float(rot_a)
                best_rotation_b = float(rot_b)
    
    return min_cost, best_rotation_a, best_rotation_b


def compute_all_pairwise_costs(
    pieces: List[Piece],
    side: str = 'right',
    method: MatchingMethod = DEFAULT_METHOD
) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """Compute matching costs between all pairs of pieces.
    
    Args:
        pieces: List of puzzle pieces
        side: Which adjacency to consider ('right' or 'bottom')
        method: Matching method to use
        
    Returns:
        Dictionary mapping (piece_i_id, piece_j_id) to (cost, rot_i, rot_j)
    """
    n = len(pieces)
    costs = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                cost, rot_i, rot_j = find_best_match(pieces[i], pieces[j], side, method)
                costs[(pieces[i].id, pieces[j].id)] = (cost, rot_i, rot_j)
    
    return costs


def build_cost_matrix(
    pieces: List[Piece],
    method: MatchingMethod = DEFAULT_METHOD
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[float, float, str]]]:
    """Build a cost matrix for MST computation.
    
    Combines costs for both horizontal (right-left) and vertical (bottom-top)
    adjacencies into a single symmetric matrix.
    
    Args:
        pieces: List of puzzle pieces
        method: Matching method to use
        
    Returns:
        Tuple of:
            - Cost matrix (n x n) where entry [i,j] is min cost between pieces i and j
            - Dictionary mapping (i, j) to (best_rotation_i, best_rotation_j, direction)
    """
    n = len(pieces)
    cost_matrix = np.full((n, n), np.inf)
    rotation_info = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Try both horizontal and vertical adjacency
                cost_h, rot_i_h, rot_j_h = find_best_match(pieces[i], pieces[j], 'right', method)
                cost_v, rot_i_v, rot_j_v = find_best_match(pieces[i], pieces[j], 'bottom', method)
                
                # Use minimum cost
                if cost_h <= cost_v:
                    cost_matrix[i, j] = cost_h
                    rotation_info[(i, j)] = (rot_i_h, rot_j_h, 'horizontal')
                else:
                    cost_matrix[i, j] = cost_v
                    rotation_info[(i, j)] = (rot_i_v, rot_j_v, 'vertical')
    
    return cost_matrix, rotation_info
