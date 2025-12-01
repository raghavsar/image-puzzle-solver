"""MST-based grid solver for puzzle reconstruction."""

from typing import List, Tuple, Dict, Optional
from collections import deque
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from ..utils.piece import Piece
from .edge_matcher import find_best_match, ROTATIONS, MatchingMethod


def solve_grid(
    pieces: List[Piece],
    grid_rows: int,
    grid_cols: int,
    verbose: bool = False,
    use_fast_matching: bool = True
) -> List[Piece]:
    """Solve the puzzle by finding optimal piece placements.
    
    Uses MST-based reconstruction:
    1. Build complete graph with edge weights = matching costs
    2. Compute MST to find most likely adjacencies
    3. BFS from a root to assign grid positions and rotations
    
    Args:
        pieces: List of puzzle pieces
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        verbose: Whether to print progress information
        use_fast_matching: Use faster SSD matching for large puzzles (>50 pieces)
        
    Returns:
        List of pieces with solved_center and solved_rotation set
    """
    n = len(pieces)
    expected_count = grid_rows * grid_cols
    
    if n != expected_count:
        raise ValueError(
            f"Piece count ({n}) doesn't match grid dimensions ({grid_rows}x{grid_cols}={expected_count})"
        )
    
    # For large puzzles, use faster SSD matching instead of histogram
    method = MatchingMethod.SSD if (use_fast_matching and n > 50) else MatchingMethod.HISTOGRAM
    if verbose:
        print(f"  Using {method.value} matching for {n} pieces...")
    
    # Build cost matrix for MST
    cost_matrix, adjacency_info = _build_adjacency_graph(pieces, verbose=verbose, method=method)
    
    # Compute MST
    mst = minimum_spanning_tree(csr_matrix(cost_matrix))
    mst_array = mst.toarray()
    
    # Make MST symmetric for easier traversal
    mst_symmetric = mst_array + mst_array.T
    
    # Find grid layout using BFS
    grid_positions, rotations = _assign_grid_positions(
        pieces, mst_symmetric, adjacency_info, grid_rows, grid_cols
    )
    
    # Calculate piece size for positioning
    piece_width, piece_height = _estimate_piece_size(pieces)
    
    # Set solved positions and rotations
    for piece in pieces:
        pos = grid_positions.get(piece.id)
        if pos is not None:
            row, col = pos
            # Calculate center position in solved image
            center_x = col * piece_width + piece_width / 2
            center_y = row * piece_height + piece_height / 2
            piece.solved_center = (center_x, center_y)
            piece.solved_rotation = rotations.get(piece.id, 0.0)
    
    return pieces


def _build_adjacency_graph(
    pieces: List[Piece],
    verbose: bool = False,
    method: MatchingMethod = MatchingMethod.SSD
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict]]:
    """Build a complete graph of piece adjacencies.
    
    Args:
        pieces: List of puzzle pieces
        verbose: Whether to print progress
        method: Matching method to use
        
    Returns:
        Tuple of:
            - Cost matrix (n x n)
            - Adjacency info dict mapping (i, j) to adjacency details
    """
    n = len(pieces)
    cost_matrix = np.full((n, n), np.inf)
    adjacency_info = {}
    
    total_pairs = n * (n - 1) // 2
    pair_count = 0
    last_progress = -1
    
    for i in range(n):
        for j in range(i + 1, n):
            pair_count += 1
            
            # Progress reporting
            if verbose:
                progress = int(100 * pair_count / total_pairs)
                if progress >= last_progress + 5:  # Report every 5%
                    print(f"  Matching progress: {progress}% ({pair_count}/{total_pairs} pairs)", end='\r')
                    sys.stdout.flush()
                    last_progress = progress
            
            # Test horizontal adjacency (i on left, j on right)
            cost_h, rot_i_h, rot_j_h = find_best_match(pieces[i], pieces[j], 'right', method)
            
            # Test vertical adjacency (i on top, j on bottom)
            cost_v, rot_i_v, rot_j_v = find_best_match(pieces[i], pieces[j], 'bottom', method)
            
            # Use minimum cost
            if cost_h <= cost_v:
                cost_matrix[i, j] = cost_h
                cost_matrix[j, i] = cost_h
                adjacency_info[(i, j)] = {
                    'direction': 'horizontal',
                    'rot_i': rot_i_h,
                    'rot_j': rot_j_h,
                    'cost': cost_h
                }
                adjacency_info[(j, i)] = {
                    'direction': 'horizontal_reverse',
                    'rot_i': rot_j_h,
                    'rot_j': rot_i_h,
                    'cost': cost_h
                }
            else:
                cost_matrix[i, j] = cost_v
                cost_matrix[j, i] = cost_v
                adjacency_info[(i, j)] = {
                    'direction': 'vertical',
                    'rot_i': rot_i_v,
                    'rot_j': rot_j_v,
                    'cost': cost_v
                }
                adjacency_info[(j, i)] = {
                    'direction': 'vertical_reverse',
                    'rot_i': rot_j_v,
                    'rot_j': rot_i_v,
                    'cost': cost_v
                }
    
    if verbose:
        print(f"  Matching progress: 100% ({total_pairs}/{total_pairs} pairs)")
    
    return cost_matrix, adjacency_info


def _assign_grid_positions(
    pieces: List[Piece],
    mst: np.ndarray,
    adjacency_info: Dict[Tuple[int, int], Dict],
    grid_rows: int,
    grid_cols: int
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, float]]:
    """Assign grid positions to pieces using BFS on MST.
    
    Args:
        pieces: List of puzzle pieces
        mst: Symmetric MST adjacency matrix
        adjacency_info: Dictionary with adjacency details
        grid_rows: Number of rows
        grid_cols: Number of columns
        
    Returns:
        Tuple of:
            - Dictionary mapping piece_id to (row, col)
            - Dictionary mapping piece_id to rotation
    """
    n = len(pieces)
    
    # Start BFS from piece 0, place it at center of grid
    start_piece = 0
    start_row = grid_rows // 2
    start_col = grid_cols // 2
    
    # Track assignments
    grid_positions = {start_piece: (start_row, start_col)}
    rotations = {start_piece: 0.0}
    
    # BFS queue
    queue = deque([start_piece])
    visited = {start_piece}
    
    # Grid occupancy (row, col) -> piece_id
    grid = {(start_row, start_col): start_piece}
    
    while queue:
        current = queue.popleft()
        current_row, current_col = grid_positions[current]
        current_rotation = rotations[current]
        
        # Find neighbors in MST
        for neighbor in range(n):
            if neighbor not in visited and mst[current, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)
                
                # Get adjacency info
                info = adjacency_info.get((current, neighbor), {})
                direction = info.get('direction', 'horizontal')
                
                # Determine relative position
                if direction in ['horizontal', 'horizontal_reverse']:
                    # Neighbor is to the right or left
                    if direction == 'horizontal':
                        new_row, new_col = current_row, current_col + 1
                    else:
                        new_row, new_col = current_row, current_col - 1
                else:
                    # Neighbor is below or above
                    if direction == 'vertical':
                        new_row, new_col = current_row + 1, current_col
                    else:
                        new_row, new_col = current_row - 1, current_col
                
                # Handle out-of-bounds by finding nearest free cell
                new_row, new_col = _find_nearest_free_cell(
                    grid, new_row, new_col, grid_rows, grid_cols
                )
                
                grid_positions[neighbor] = (new_row, new_col)
                grid[(new_row, new_col)] = neighbor
                rotations[neighbor] = info.get('rot_j', 0.0)
    
    # Normalize positions to start from (0, 0)
    min_row = min(pos[0] for pos in grid_positions.values())
    min_col = min(pos[1] for pos in grid_positions.values())
    
    normalized_positions = {
        piece_id: (row - min_row, col - min_col)
        for piece_id, (row, col) in grid_positions.items()
    }
    
    return normalized_positions, rotations


def _find_nearest_free_cell(
    grid: Dict[Tuple[int, int], int],
    target_row: int,
    target_col: int,
    max_rows: int,
    max_cols: int
) -> Tuple[int, int]:
    """Find the nearest unoccupied cell to the target position.
    
    Args:
        grid: Current grid occupancy
        target_row: Desired row
        target_col: Desired column
        max_rows: Maximum rows
        max_cols: Maximum columns
        
    Returns:
        Tuple of (row, col) for the nearest free cell
    """
    # If target is free, use it
    if (target_row, target_col) not in grid:
        return target_row, target_col
    
    # BFS to find nearest free cell
    queue = deque([(target_row, target_col, 0)])
    visited = {(target_row, target_col)}
    
    while queue:
        row, col, dist = queue.popleft()
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            
            if (new_row, new_col) not in visited:
                visited.add((new_row, new_col))
                
                if (new_row, new_col) not in grid:
                    return new_row, new_col
                
                queue.append((new_row, new_col, dist + 1))
    
    # Fallback: return target anyway
    return target_row, target_col


def _estimate_piece_size(pieces: List[Piece]) -> Tuple[int, int]:
    """Estimate the standard piece size from the pieces.
    
    Args:
        pieces: List of puzzle pieces
        
    Returns:
        Tuple of (width, height)
    """
    if not pieces:
        return 100, 100
    
    # Use median size to be robust to outliers
    widths = [p.width for p in pieces]
    heights = [p.height for p in pieces]
    
    return int(np.median(widths)), int(np.median(heights))


def compute_grid_dimensions(n: int, aspect_ratio: float = 1.0) -> Tuple[int, int]:
    """Compute grid dimensions for N pieces.
    
    Args:
        n: Number of pieces
        aspect_ratio: Desired width/height ratio
        
    Returns:
        Tuple of (rows, cols)
    """
    # Try perfect square first
    sqrt_n = int(np.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return sqrt_n, sqrt_n
    
    # Find factor pair closest to sqrt(n)
    best_rows, best_cols = 1, n
    min_diff = abs(n - 1)
    
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            rows, cols = i, n // i
            diff = abs(cols / rows - aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_rows, best_cols = rows, cols
    
    return best_rows, best_cols
