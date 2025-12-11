"""Greedy grid solver for puzzle reconstruction.

This solver uses a greedy best-first approach:
1. Find the best matching piece for the top-left corner
2. Build the first row by finding best right-edge matches
3. Build subsequent rows by finding best bottom-edge matches
4. Each piece placement tests all 4 rotations to find the best fit
"""

from typing import List, Tuple, Dict, Optional, Set, Iterable
import numpy as np

from ..utils.piece import Piece
from .edge_matcher import compute_edge_cost, MatchingMethod, ROTATIONS


def solve_grid_greedy(
    pieces: List[Piece],
    grid_rows: int,
    grid_cols: int,
    verbose: bool = False,
    method: MatchingMethod = MatchingMethod.SSD,
    rotations: Iterable[float] = ROTATIONS,
    anchor_piece: int = 0
) -> List[Piece]:
    """Solve the puzzle by greedy row-by-row assembly.
    
    Algorithm:
    1. Select a starting piece (use piece with most distinctive corners)
    2. Build first row: for each position, find unplaced piece with best right-edge match
    3. Build remaining rows: for each position, find best piece matching both
       left neighbor (if exists) and top neighbor
    4. Each piece is tested at all 4 rotations
    
    Args:
        pieces: List of puzzle pieces
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        verbose: Whether to print progress information
        method: Edge matching method to use
        
    Returns:
        List of pieces with solved_center and solved_rotation set
    """
    n = len(pieces)
    expected_count = grid_rows * grid_cols
    
    if n != expected_count:
        raise ValueError(
            f"Piece count ({n}) doesn't match grid dimensions ({grid_rows}x{grid_cols}={expected_count})"
        )
    
    rotations_to_try = tuple(rotations)

    if verbose:
        print(f"  Solving {grid_rows}x{grid_cols} grid with greedy algorithm...")
        print(f"  Using {method.value} matching...")
        rot_str = ", ".join(str(r) for r in rotations_to_try)
        print(f"  Rotations considered: {rot_str}")
        print(f"  Anchor piece index: {anchor_piece}")
    
    # Build the grid
    grid: List[List[Optional[Tuple[int, float]]]] = [[None] * grid_cols for _ in range(grid_rows)]
    used_pieces: Set[int] = set()
    
    # Precompute all pairwise matching costs for efficiency
    if verbose:
        print("  Precomputing pairwise matching costs...")
    
    # Cache: (piece_i, rot_i, piece_j, rot_j, side) -> cost
    cost_cache: Dict[Tuple[int, float, int, float, str], float] = {}
    
    def get_match_cost(piece_i: int, rot_i: float, piece_j: int, rot_j: float, side: str) -> float:
        """Get or compute matching cost between two pieces."""
        key = (piece_i, rot_i, piece_j, rot_j, side)
        if key not in cost_cache:
            cost_cache[key] = compute_edge_cost(
                pieces[piece_i], pieces[piece_j], side, rot_i, rot_j, method
            )
        return cost_cache[key]
    
    def find_best_piece_for_position(
        row: int, col: int,
        left_piece: Optional[Tuple[int, float]],
        top_piece: Optional[Tuple[int, float]]
    ) -> Tuple[int, float, float]:
        """Find the best unplaced piece for a given grid position.
        
        Args:
            row, col: Grid position
            left_piece: (piece_id, rotation) of left neighbor, or None
            top_piece: (piece_id, rotation) of top neighbor, or None
            
        Returns:
            (piece_id, rotation, cost) of best match
        """
        best_piece = -1
        best_rotation = 0.0
        best_cost = float('inf')
        
        for piece_idx in range(n):
            if piece_idx in used_pieces:
                continue
            
            for rotation in rotations_to_try:
                rot = float(rotation)
                total_cost = 0.0
                
                # Match against left neighbor
                if left_piece is not None:
                    left_idx, left_rot = left_piece
                    # Left piece's right edge should match this piece's left edge
                    cost = get_match_cost(left_idx, left_rot, piece_idx, rot, 'right')
                    total_cost += cost
                
                # Match against top neighbor
                if top_piece is not None:
                    top_idx, top_rot = top_piece
                    # Top piece's bottom edge should match this piece's top edge
                    cost = get_match_cost(top_idx, top_rot, piece_idx, rot, 'bottom')
                    total_cost += cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_piece = piece_idx
                    best_rotation = rot
        
        return best_piece, best_rotation, best_cost
    
    # Step 1: Select starting piece for top-left
    # Use piece 0 at rotation 0 as the anchor (any piece works as reference)
    if verbose:
        print("  Placing pieces in grid...")
    
    # For translation-only puzzles, we can start with any rotation
    # For rotation puzzles, we need to find the correct orientation
    
    # Place top-left corner - try to find a piece that has good edge characteristics
    start_piece_idx = anchor_piece
    start_rotation = float(rotations_to_try[0]) if rotations_to_try else 0.0
    
    grid[0][0] = (start_piece_idx, start_rotation)
    used_pieces.add(start_piece_idx)
    
    # Step 2: Build first row (left to right)
    for col in range(1, grid_cols):
        left_piece = grid[0][col - 1]
        piece_idx, rotation, cost = find_best_piece_for_position(0, col, left_piece, None)
        
        if piece_idx < 0:
            raise RuntimeError(f"Failed to find piece for position (0, {col})")
        
        grid[0][col] = (piece_idx, rotation)
        used_pieces.add(piece_idx)
        
        if verbose:
            print(f"    Row 0, Col {col}: Piece {piece_idx} @ {rotation}° (cost: {cost:.2f})")
    
    # Step 3: Build remaining rows
    for row in range(1, grid_rows):
        for col in range(grid_cols):
            left_piece = grid[row][col - 1] if col > 0 else None
            top_piece = grid[row - 1][col]
            
            piece_idx, rotation, cost = find_best_piece_for_position(row, col, left_piece, top_piece)
            
            if piece_idx < 0:
                raise RuntimeError(f"Failed to find piece for position ({row}, {col})")
            
            grid[row][col] = (piece_idx, rotation)
            used_pieces.add(piece_idx)
            
            if verbose:
                print(f"    Row {row}, Col {col}: Piece {piece_idx} @ {rotation}° (cost: {cost:.2f})")
    
    if verbose:
        print("  Grid assembly complete!")
        print_grid(grid)
    
    # Step 4: Convert grid to solved positions
    piece_width = pieces[0].width
    piece_height = pieces[0].height
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            piece_idx, rotation = grid[row][col]
            center_x = col * piece_width + piece_width / 2
            center_y = row * piece_height + piece_height / 2
            pieces[piece_idx].solved_center = (center_x, center_y)
            pieces[piece_idx].solved_rotation = rotation
    
    return pieces


def print_grid(grid: List[List[Optional[Tuple[int, float]]]]) -> None:
    """Print the grid layout for debugging."""
    print("  Grid layout:")
    for row in grid:
        row_str = " | ".join(
            f"P{p:2d}@{r:3.0f}°" if (p, r) is not None else "  ----  "
            for p, r in row
        )
        print(f"    [{row_str}]")


def solve_grid_with_global_rotation(
    pieces: List[Piece],
    grid_rows: int,
    grid_cols: int,
    verbose: bool = False,
    method: MatchingMethod = MatchingMethod.SSD,
    anchor_piece: int = 0
) -> List[Piece]:
    """Solve puzzle trying all 4 global rotations of the anchor piece.
    
    This helps handle cases where the entire puzzle might be rotated.
    Returns the solution with the lowest total matching cost.
    
    Args:
        pieces: List of puzzle pieces
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        verbose: Whether to print progress information
        method: Edge matching method to use
        
    Returns:
        List of pieces with solved_center and solved_rotation set
    """
    from copy import deepcopy
    
    best_pieces = None
    best_cost = float('inf')
    best_anchor_rotation = 0
    
    for anchor_rotation in ROTATIONS:
        if verbose:
            print(f"\n  Trying anchor rotation: {anchor_rotation}°")
        
        # Create a copy of pieces
        pieces_copy = [
            Piece(
                id=p.id,
                image=p.image.copy(),
                initial_center=p.initial_center,
                initial_rotation=p.initial_rotation
            )
            for p in pieces
        ]
        
        # Solve with this anchor rotation
        try:
            solved = _solve_with_anchor_rotation(
                pieces_copy, grid_rows, grid_cols,
                anchor_rotation=float(anchor_rotation),
                verbose=verbose,
                method=method,
                anchor_piece=anchor_piece
            )
            
            # Compute total cost
            total_cost = _compute_solution_cost(solved, grid_rows, grid_cols, method)
            
            if verbose:
                print(f"  Total cost for anchor {anchor_rotation}°: {total_cost:.2f}")
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_pieces = solved
                best_anchor_rotation = anchor_rotation
        
        except Exception as e:
            if verbose:
                print(f"  Failed with anchor {anchor_rotation}°: {e}")
    
    if best_pieces is None:
        raise RuntimeError("All anchor rotations failed")
    
    if verbose:
        print(f"\n  Best solution: anchor at {best_anchor_rotation}° with cost {best_cost:.2f}")
    
    # Copy best solution to original pieces
    for orig, solved in zip(pieces, best_pieces):
        orig.solved_center = solved.solved_center
        orig.solved_rotation = solved.solved_rotation
    
    return pieces


def _solve_with_anchor_rotation(
    pieces: List[Piece],
    grid_rows: int,
    grid_cols: int,
    anchor_rotation: float,
    verbose: bool,
    method: MatchingMethod,
    anchor_piece: int = 0
) -> List[Piece]:
    """Internal solver with specific anchor rotation."""
    n = len(pieces)
    
    grid: List[List[Optional[Tuple[int, float]]]] = [[None] * grid_cols for _ in range(grid_rows)]
    used_pieces: Set[int] = set()
    
    # Cache for matching costs
    cost_cache: Dict[Tuple[int, float, int, float, str], float] = {}
    
    def get_match_cost(piece_i: int, rot_i: float, piece_j: int, rot_j: float, side: str) -> float:
        key = (piece_i, rot_i, piece_j, rot_j, side)
        if key not in cost_cache:
            cost_cache[key] = compute_edge_cost(
                pieces[piece_i], pieces[piece_j], side, rot_i, rot_j, method
            )
        return cost_cache[key]
    
    def find_best_piece(
        left_piece: Optional[Tuple[int, float]],
        top_piece: Optional[Tuple[int, float]]
    ) -> Tuple[int, float, float]:
        best_piece = -1
        best_rotation = 0.0
        best_cost = float('inf')
        
        for piece_idx in range(n):
            if piece_idx in used_pieces:
                continue
            
            for rotation in ROTATIONS:
                rot = float(rotation)
                total_cost = 0.0
                
                if left_piece is not None:
                    left_idx, left_rot = left_piece
                    cost = get_match_cost(left_idx, left_rot, piece_idx, rot, 'right')
                    total_cost += cost
                
                if top_piece is not None:
                    top_idx, top_rot = top_piece
                    cost = get_match_cost(top_idx, top_rot, piece_idx, rot, 'bottom')
                    total_cost += cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_piece = piece_idx
                    best_rotation = rot
        
        return best_piece, best_rotation, best_cost
    
    # Place anchor
    grid[0][0] = (anchor_piece, anchor_rotation)
    used_pieces.add(anchor_piece)
    
    # Build first row
    for col in range(1, grid_cols):
        left_piece = grid[0][col - 1]
        piece_idx, rotation, _ = find_best_piece(left_piece, None)
        if piece_idx < 0:
            raise RuntimeError(f"No piece for (0, {col})")
        grid[0][col] = (piece_idx, rotation)
        used_pieces.add(piece_idx)
    
    # Build remaining rows
    for row in range(1, grid_rows):
        for col in range(grid_cols):
            left_piece = grid[row][col - 1] if col > 0 else None
            top_piece = grid[row - 1][col]
            piece_idx, rotation, _ = find_best_piece(left_piece, top_piece)
            if piece_idx < 0:
                raise RuntimeError(f"No piece for ({row}, {col})")
            grid[row][col] = (piece_idx, rotation)
            used_pieces.add(piece_idx)
    
    # Set solved positions
    piece_width = pieces[0].width
    piece_height = pieces[0].height
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            piece_idx, rotation = grid[row][col]
            center_x = col * piece_width + piece_width / 2
            center_y = row * piece_height + piece_height / 2
            pieces[piece_idx].solved_center = (center_x, center_y)
            pieces[piece_idx].solved_rotation = rotation
    
    return pieces


def _compute_solution_cost(
    pieces: List[Piece],
    grid_rows: int,
    grid_cols: int,
    method: MatchingMethod
) -> float:
    """Compute total matching cost of a solution."""
    # Build grid from solved positions
    piece_width = pieces[0].width
    
    grid = {}
    for p in pieces:
        if p.solved_center:
            col = int(p.solved_center[0] / piece_width)
            row = int(p.solved_center[1] / piece_width)
            grid[(row, col)] = (p.id, p.solved_rotation)
    
    total_cost = 0.0
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if (row, col) not in grid:
                continue
            
            piece_idx, rotation = grid[(row, col)]
            
            # Check right neighbor
            if col + 1 < grid_cols and (row, col + 1) in grid:
                right_idx, right_rot = grid[(row, col + 1)]
                cost = compute_edge_cost(
                    pieces[piece_idx], pieces[right_idx],
                    'right', rotation, right_rot, method
                )
                total_cost += cost
            
            # Check bottom neighbor
            if row + 1 < grid_rows and (row + 1, col) in grid:
                bottom_idx, bottom_rot = grid[(row + 1, col)]
                cost = compute_edge_cost(
                    pieces[piece_idx], pieces[bottom_idx],
                    'bottom', rotation, bottom_rot, method
                )
                total_cost += cost
    
    return total_cost


def compute_grid_dimensions(n: int, aspect_ratio: float = 1.0) -> Tuple[int, int]:
    """Compute grid dimensions for N pieces.
    
    Args:
        n: Number of pieces
        aspect_ratio: Desired width/height ratio
        
    Returns:
        Tuple of (rows, cols)
    """
    sqrt_n = int(np.sqrt(n))
    if sqrt_n * sqrt_n == n:
        return sqrt_n, sqrt_n
    
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
