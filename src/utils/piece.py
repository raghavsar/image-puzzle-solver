"""Piece dataclass for representing puzzle pieces."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class Piece:
    """Represents a single puzzle piece.
    
    Attributes:
        id: Unique identifier for the piece
        image: The piece image (BGR format for OpenCV) - already unrotated/axis-aligned
        initial_center: Initial (x, y) center position in the scrambled image
        initial_rotation: Initial rotation angle in degrees (how much piece was rotated in scrambled image)
        solved_center: Final (x, y) center position in the solved image
        solved_rotation: Final rotation angle in degrees (typically 0)
        mask: Binary mask for the piece (for alpha-blended compositing)
    """
    id: int
    image: np.ndarray
    initial_center: Tuple[float, float]
    initial_rotation: float = 0.0
    solved_center: Optional[Tuple[float, float]] = None
    solved_rotation: float = 0.0
    mask: Optional[np.ndarray] = None
    
    def get_rotated_image(self, angle: float) -> np.ndarray:
        """Get the piece image rotated by the specified angle.
        
        Args:
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated image as numpy array
        """
        if angle == 0:
            return self.image.copy()
        
        h, w = self.image.shape[:2]
        center = (w / 2, h / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Perform rotation
        rotated = cv2.warpAffine(self.image, rotation_matrix, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        return rotated
    
    def get_rotated_mask(self, angle: float) -> np.ndarray:
        """Get the piece mask rotated by the specified angle.
        
        Args:
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated mask as numpy array
        """
        mask = self.get_mask()
        
        if angle == 0:
            return mask.copy()
        
        h, w = mask.shape[:2]
        center = (w / 2, h / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Perform rotation
        rotated = cv2.warpAffine(mask, rotation_matrix, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return rotated
    
    def get_mask(self) -> np.ndarray:
        """Get or create the piece mask.
        
        Returns:
            Binary mask where piece pixels are 255, background is 0
        """
        if self.mask is not None:
            return self.mask
        
        # Create mask from non-zero pixels
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        return mask
    
    def get_edge(self, side: str, rotation: float = 0.0, strip_width: int = 5) -> np.ndarray:
        """Extract edge pixels from a specific side of the piece.
        
        Args:
            side: Which edge to extract ('top', 'bottom', 'left', 'right')
            rotation: Rotation angle to apply before extracting edge
            strip_width: Number of pixel rows/columns to extract (default 5)
            
        Returns:
            1D array of edge pixel values (flattened RGB)
        """
        # Get potentially rotated image
        img = self.get_rotated_image(rotation) if rotation != 0 else self.image
        
        # Map side to actual edge based on rotation
        # For simplicity, extract edge from the specified side of the (rotated) image
        h, w = img.shape[:2]
        
        # Clamp strip_width to available size
        strip_width = min(strip_width, min(h, w) // 2)
        
        if side == 'top':
            edge = img[:strip_width, :, :]
        elif side == 'bottom':
            edge = img[h - strip_width:, :, :]
        elif side == 'left':
            edge = img[:, :strip_width, :]
        elif side == 'right':
            edge = img[:, w - strip_width:, :]
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'top', 'bottom', 'left', or 'right'")
        
        return edge.flatten().astype(np.float64)
    
    @property
    def width(self) -> int:
        """Get piece width."""
        return self.image.shape[1]
    
    @property
    def height(self) -> int:
        """Get piece height."""
        return self.image.shape[0]
    
    def __repr__(self) -> str:
        return (f"Piece(id={self.id}, size=({self.width}x{self.height}), "
                f"initial_center={self.initial_center}, initial_rotation={self.initial_rotation})")
