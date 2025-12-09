import cv2
import os
import numpy as np
import imageio
from typing import List, Dict, Tuple

from Tile import Tile
from Pixel import Pixel

font = cv2.FONT_HERSHEY_COMPLEX

CANVAS_SIZE = (800, 800)  # Unified canvas size


def generate_puzzle_animation(
        tiles: List[Tile],
        original_img: np.ndarray,
        frame_count: int = 15,
        output_filename: str = "puzzle_solution.gif",
):
    """
    Generates an animated GIF based on the initial and final information of the Tile objects.

    Args:
        tiles: A list of Tile objects containing initial/final position and rotation.
        original_img: The original BGR image used to crop the puzzle pieces.
        frame_count: The total number of frames for the animation.
        output_filename: The name of the output GIF file.
    """

    frames = []
    H, W = CANVAS_SIZE  # Unified canvas height and width

    # Pre-crop all tile image parts
    tile_images = [t.image for t in tiles if t.image is not None]

    if not tile_images:
        print("Error: No cropped images available for animation. Ensure image is set in simulate_solve_puzzle.")
        return

    # print(f"Starting animation generation with {frame_count} frames...")

    for f in range(1, frame_count + 1):
        # Calculate interpolation factor, from 0 to 1
        # alpha = f / frame_count
        alpha = (f - 1) / (frame_count - 1)

        # Create a blank canvas (black background)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        for tile in tiles:
            # Skip tiles without image data
            if tile.image is None:
                continue

            tile_img = tile.image
            img_h, img_w, _ = tile_img.shape

            # --- 1. Calculate current frame position and angle (Linear Interpolation) ---

            # Get initial/final info from Tile object
            x_start, y_start = tile.initial_position
            x_end, y_end = tile.final_position
            angle_start = tile.initial_rotation
            angle_end = tile.final_rotation

            # Position interpolation
            x_curr = int(x_start * (1 - alpha) + x_end * alpha)
            y_curr = int(y_start * (1 - alpha) + y_end * alpha)

            # Angle interpolation
            angle_curr = angle_start * (1 - alpha) + angle_end * alpha

            # --- 2. Apply rotation and affine transformation ---

            center = (img_w / 2, img_h / 2)
            M_rot = cv2.getRotationMatrix2D(center, angle_curr, 1.0)
            rotated_tile = cv2.warpAffine(tile_img, M_rot, (img_w, img_h),
                                          borderValue=(0,0,0))

            # --- 3. Place the puzzle piece onto the canvas ---

            x1, y1 = x_curr, y_curr
            x2, y2 = x_curr + img_w, y_curr + img_h

            # Boundary check
            x2 = min(x2, W)
            y2 = min(y2, H)

            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                source_img = rotated_tile[0:y2 - y1, 0:x2 - x1]
                # Place onto the canvas
                canvas[y1:y2, x1:x2] = source_img

        # Convert to RGB (imageio requires RGB order)
        frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Write GIF (using fps to control speed)
    imageio.mimsave(output_filename, frames, fps=15)
    print(f"\nAnimation successfully generated and saved as {output_filename}")
