# solver.py
import numpy as np
import cv2
from Tile import Tile
import math

def extract_edge_features(tile):
    """Extract edge components matching."""
    img = tile.image
    h, w = img.shape[:2]

    edges = {
        'top': img[0, :, :].astype(np.float32),
        'bottom': img[h-1, :, :].astype(np.float32),
        'left': img[:, 0, :].astype(np.float32),
        'right': img[:, w-1, :].astype(np.float32)
    }
    return edges

def edge_similarity(edge1, edge2):
    """Use mean squared error for similarity."""
    if edge1.shape != edge2.shape:
        edge2 = cv2.resize(edge2, (edge1.shape[1], edge1.shape[0]))
    return -np.mean((edge1 - edge2) ** 2)

def compute_all_matches(tiles):
    """Compute all the edges"""
    n = len(tiles)
    features = [extract_edge_features(tile) for tile in tiles]
    matches = {}

    for i in range(n):
        matches[i] = {}
        for j in range(n):
            if i == j:
                continue
            best_score = -np.inf
            best_dir = None
            for dir1, edge1 in features[i].items():
                for dir2, edge2 in features[j].items():
                    if (dir1 == 'top' and dir2 == 'bottom') or \
                       (dir1 == 'bottom' and dir2 == 'top') or \
                       (dir1 == 'left' and dir2 == 'right') or \
                       (dir1 == 'right' and dir2 == 'left'):
                        score = edge_similarity(edge1, edge2)
                        if score > best_score:
                            best_score = score
                            best_dir = (dir1, dir2)
            matches[i][j] = (best_score, best_dir)
    return matches

def assemble_puzzle_global(tiles, matches):
    N = len(tiles)
    placed = {}
    remaining = set(range(N))

    # --- Find the pair with highest edge match ---
    best_pair = None
    best_score = -np.inf
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            score, dirs = matches[i][j]
            if score > best_score:
                best_score = score
                best_pair = (i, j, dirs)

    # Place the first two tiles based on best pair
    i, j, dirs = best_pair
    placed[i] = (0, 0)
    # Place j relative to i
    if dirs[0] == 'right':
        placed[j] = (1, 0)
    elif dirs[0] == 'left':
        placed[j] = (-1, 0)
    elif dirs[0] == 'bottom':
        placed[j] = (0, 1)
    elif dirs[0] == 'top':
        placed[j] = (0, -1)

    remaining.remove(i)
    remaining.remove(j)

    while remaining:
        best_tile = None
        best_pos = None
        best_score = -np.inf

        for idx in remaining:
            for pid, ppos in placed.items():
                score, dirs = matches[pid][idx]
                if dirs is None:
                    continue
                x, y = ppos
                if dirs[0] == 'right':
                    pos = (x+1, y)
                elif dirs[0] == 'left':
                    pos = (x-1, y)
                elif dirs[0] == 'bottom':
                    pos = (x, y+1)
                elif dirs[0] == 'top':
                    pos = (x, y-1)
                # Avoid overlapping
                if pos in placed.values():
                    continue
                if score > best_score:
                    best_score = score
                    best_tile = idx
                    best_pos = pos

        # --- Fallback if no tile found ---
        if best_tile is None:
            # pick any remaining tile
            best_tile = remaining.pop()
            # place it next to any already placed tile (first one)
            first_pos = list(placed.values())[0]
            best_pos = (first_pos[0] + 1, first_pos[1])
        else:
            remaining.remove(best_tile)

        placed[best_tile] = best_pos

    # --- Normalize positions to top-left (0,0) ---
    xs = [p[0] for p in placed.values()]
    ys = [p[1] for p in placed.values()]
    min_x, min_y = min(xs), min(ys)
    normalized_positions = {}
    for idx, (x, y) in placed.items():
        normalized_positions[idx] = (x - min_x, y - min_y)
        tiles[idx].position = normalized_positions[idx]

    # --- Build grid for convenience ---
    max_x = max(x for x, y in normalized_positions.values())
    max_y = max(y for x, y in normalized_positions.values())
    grid = [[None for _ in range(max_x+1)] for _ in range(max_y+1)]
    for idx, (x, y) in normalized_positions.items():
        grid[y][x] = tiles[idx]

    # --- Build grid for convenience ---
    max_x = max(x for x, y in normalized_positions.values())
    max_y = max(y for x, y in normalized_positions.values())
    grid = [[None for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for idx, (x, y) in normalized_positions.items():
        grid[y][x] = tiles[idx]

    # --- Compute cell pixel sizes per column/row to handle varying tile sizes ---
    col_widths = {}
    row_heights = {}
    for idx, (col, row) in normalized_positions.items():
        h, w = tiles[idx].image.shape[:2]
        col_widths[col] = max(col_widths.get(col, 0), w)
        row_heights[row] = max(row_heights.get(row, 0), h)

    # Ensure contiguous columns/rows (0..max_col / 0..max_row)
    cols = list(range(0, max_x + 1))
    rows = list(range(0, max_y + 1))

    # margin around whole assembled image (in pixels)
    margin = 20
    # compute total pixel dimensions
    total_w = sum(col_widths[c] for c in cols)
    total_h = sum(row_heights[r] for r in rows)

    # compute top-left pixel for each column and row
    col_x = {}
    cur_x = margin
    for c in cols:
        col_x[c] = cur_x
        cur_x += col_widths[c]

    row_y = {}
    cur_y = margin
    for r in rows:
        row_y[r] = cur_y
        cur_y += row_heights[r]

    # Now set final_position (pixel coordinates, top-left) for every tile,
    # center each tile inside its cell if its size < cell size.
    for idx, (col, row) in normalized_positions.items():
        cell_w = col_widths[col]
        cell_h = row_heights[row]
        tile_h, tile_w = tiles[idx].image.shape[:2]

        tx = col_x[col] + (cell_w - tile_w) // 2
        ty = row_y[row] + (cell_h - tile_h) // 2

        tiles[idx].final_position = (int(tx), int(ty))

        # set final_rotation if desired (default 0 -- upright)
        tiles[idx].final_rotation = 0

    return tiles, grid, max_y+1, max_x+1

def simulate_solve_puzzle(tiles, img):
    """Solve puzzle using global placement and display result (handles varying tile sizes)."""
    matches = compute_all_matches(tiles)
    assembled_tiles, grid, rows, cols = assemble_puzzle_global(tiles, matches)

    # Determine standard tile size (max height and width among all tiles)
    tile_h = max(tile.image.shape[0] for tile in assembled_tiles)
    tile_w = max(tile.image.shape[1] for tile in assembled_tiles)

    canvas_h = rows * tile_h
    canvas_w = cols * tile_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place tiles onto canvas, resizing to standard size if needed
    for tile in assembled_tiles:
        c, r = tile.position
        y_start = r * tile_h
        y_end = y_start + tile_h
        x_start = c * tile_w
        x_end = x_start + tile_w

        resized_tile = cv2.resize(tile.image, (tile_w, tile_h))
        canvas[y_start:y_end, x_start:x_end] = resized_tile

    cv2.imshow("Puzzle Solution", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return assembled_tiles
