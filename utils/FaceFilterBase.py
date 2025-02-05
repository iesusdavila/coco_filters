import cv2
import glob
import os
import numpy as np

class FaceFilter:
    """
    Base class for face filters. Handles loading assets and provides utility methods for overlaying images.
    """

    def __init__(self, assets_path):
        """
        Initializes the FaceFilter with assets from the given path.

        Args:
            assets_path (str): Path to the directory containing asset images.
            
        """
        self.assets = self.load_assets(assets_path)
        self.current_asset_idx = 0
        
    def load_assets(self, path: str) -> list:
        """
        Loads all assets from a directory.

        Args:
            path (str): Path to the directory containing asset images.

        Returns:
            list: List of loaded images.
        """
        assets = []
        for file in glob.glob(os.path.join(path, "*.*")):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img is not None:
                assets.append(img)
        return assets
    
    def apply_filter(self, frame: np.ndarray, face_landmarks: list, frame_size: tuple) -> np.ndarray:
        """
        Abstract method to apply the filter. Must be implemented in subclasses.

        Args:
            frame (numpy.ndarray): The image frame to apply the filter on.
            face_landmarks (list): List of facial landmarks.
            frame_size (tuple): Size of the frame.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclasses")

    def optimized_overlay(self, bg: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Optimized overlay using NumPy for blending an overlay image onto a background image.

        Args:
            bg (numpy.ndarray): The background image.
            overlay (numpy.ndarray): The overlay image with an alpha channel.
            x (int): The x-coordinate where the overlay should be placed.
            y (int): The y-coordinate where the overlay should be placed.

        Returns:
            numpy.ndarray: The background image with the overlay applied.
        """
        h_overlay, w_overlay = overlay.shape[:2]
        
        # Regions of interest
        y1, y2 = max(y, 0), min(y + h_overlay, bg.shape[0])
        x1, x2 = max(x, 0), min(x + w_overlay, bg.shape[1])
        
        if x1 >= x2 or y1 >= y2: 
            return bg

        # Extract alpha channels
        overlay_region = overlay[y1-y:y2-y, x1-x:x2-x]
        alpha = overlay_region[..., 3:] / 255.0
        bg_region = bg[y1:y2, x1:x2]

        # Blend images
        bg[y1:y2, x1:x2] = (bg_region * (1 - alpha) + overlay_region[..., :3] * alpha).astype(np.uint8)
        return bg
