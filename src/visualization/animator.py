"""Animation utilities for creating MP4 puzzle solution videos."""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from ..utils.piece import Piece


def create_animation(
    pieces: List[Piece],
    output_path: str = "solution.mp4",
    canvas_size: Tuple[int, int] = (800, 800),
    fps: int = 30,
    duration: float = 5.0,
    show_preview: bool = False
) -> str:
    """Create an MP4 animation showing pieces moving to their solved positions.
    
    Args:
        pieces: List of pieces with both initial and solved positions set
        output_path: Path for the output MP4 file
        canvas_size: Size of the animation canvas (width, height)
        fps: Frames per second
        duration: Animation duration in seconds
        show_preview: Whether to show a preview window
        
    Returns:
        Path to the created MP4 file
    """
    num_frames = int(fps * duration)
    width, height = canvas_size
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert Y axis to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    # Initialize piece images on the canvas
    piece_artists = []
    for piece in pieces:
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(piece.image, cv2.COLOR_BGR2RGB)
        
        # Create initial extent
        cx, cy = piece.initial_center
        h, w = piece.image.shape[:2]
        extent = [cx - w/2, cx + w/2, cy + h/2, cy - h/2]
        
        artist = ax.imshow(img_rgb, extent=extent, animated=True)
        piece_artists.append(artist)
    
    def interpolate_value(start: float, end: float, t: float) -> float:
        """Smooth interpolation between start and end values."""
        # Use ease-in-out cubic
        t = t * t * (3 - 2 * t)
        return start + (end - start) * t
    
    def interpolate_angle(start: float, end: float, t: float) -> float:
        """Interpolate angle using shortest path."""
        # Normalize angles to [0, 360)
        start = start % 360
        end = end % 360
        
        # Find shortest path
        diff = end - start
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        t = t * t * (3 - 2 * t)  # Ease-in-out
        return start + diff * t
    
    def update(frame: int):
        """Update function for animation."""
        t = frame / (num_frames - 1) if num_frames > 1 else 1.0
        
        for piece, artist in zip(pieces, piece_artists):
            if piece.solved_center is None:
                continue
            
            # Interpolate position
            ix, iy = piece.initial_center
            sx, sy = piece.solved_center
            cx = interpolate_value(ix, sx, t)
            cy = interpolate_value(iy, sy, t)
            
            # Interpolate rotation
            angle = interpolate_angle(piece.initial_rotation, piece.solved_rotation, t)
            
            # Get rotated image
            rotated_img = piece.get_rotated_image(angle)
            img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            
            # Update extent
            h, w = rotated_img.shape[:2]
            extent = [cx - w/2, cx + w/2, cy + h/2, cy - h/2]
            
            artist.set_data(img_rgb)
            artist.set_extent(extent)
        
        return piece_artists
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=True, interval=1000/fps
    )
    
    # Save to MP4
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to use ffmpeg writer
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata={'title': 'Puzzle Solution'})
        anim.save(str(output_path), writer=writer)
    except Exception as e:
        # Fallback to pillow writer for GIF
        print(f"FFmpeg not available, falling back to GIF output: {e}")
        gif_path = output_path.with_suffix('.gif')
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer)
        output_path = gif_path
    
    if show_preview:
        plt.show()
    else:
        plt.close(fig)
    
    return str(output_path)


def render_solved_image(
    pieces: List[Piece],
    canvas_size: Tuple[int, int] = (800, 800)
) -> np.ndarray:
    """Render the solved puzzle as a static image.
    
    Args:
        pieces: List of pieces with solved positions set
        canvas_size: Size of the output canvas
        
    Returns:
        Solved puzzle image as numpy array (BGR)
    """
    width, height = canvas_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    for piece in pieces:
        if piece.solved_center is None:
            continue
        
        # Get rotated image
        rotated = piece.get_rotated_image(piece.solved_rotation)
        h, w = rotated.shape[:2]
        
        # Calculate position
        cx, cy = piece.solved_center
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = x1 + w
        y2 = y1 + h
        
        # Clip to canvas bounds
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = w - max(0, x2 - width)
        src_y2 = h - max(0, y2 - height)
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(width, x2)
        dst_y2 = min(height, y2)
        
        # Only copy non-zero pixels (handle piece masks)
        piece_region = rotated[src_y1:src_y2, src_x1:src_x2]
        mask = np.any(piece_region > 0, axis=2)
        
        canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
        canvas_region[mask] = piece_region[mask]
    
    return canvas


def create_side_by_side(
    original: np.ndarray,
    solved: np.ndarray,
    output_path: str = "comparison.png"
) -> str:
    """Create a side-by-side comparison image.
    
    Args:
        original: Original scrambled image
        solved: Solved puzzle image
        output_path: Output file path
        
    Returns:
        Path to the saved comparison image
    """
    # Resize if needed to match heights
    h1, w1 = original.shape[:2]
    h2, w2 = solved.shape[:2]
    
    if h1 != h2:
        scale = h1 / h2
        solved = cv2.resize(solved, (int(w2 * scale), h1))
        h2, w2 = solved.shape[:2]
    
    # Create side-by-side image
    combined = np.zeros((h1, w1 + w2 + 20, 3), dtype=np.uint8)
    combined[:, :w1] = original
    combined[:, w1 + 20:] = solved
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Solved", (w1 + 30, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, combined)
    return output_path
