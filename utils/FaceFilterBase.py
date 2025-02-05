import cv2
import glob
import os
import numpy as np

class FaceFilter:
    def __init__(self, assets_path):
        self.assets = self.load_assets(assets_path)
        self.current_asset_idx = 0
        
    def load_assets(self, path):
        """Carga todos los assets de un directorio"""
        assets = []
        for file in glob.glob(os.path.join(path, "*.*")):
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img is not None:
                assets.append(img)
        return assets
    
    def apply_filter(self, frame, face_landmarks, frame_size):
        """Método abstracto para aplicar el filtro"""
        raise NotImplementedError("Debe implementarse en subclases")

    def optimized_overlay(self, bg, overlay, x, y):
        """Superposición vectorizada con NumPy"""
        h_overlay, w_overlay = overlay.shape[:2]
        
        # Regiones de interés
        y1, y2 = max(y, 0), min(y + h_overlay, bg.shape[0])
        x1, x2 = max(x, 0), min(x + w_overlay, bg.shape[1])
        
        if x1 >= x2 or y1 >= y2: 
            return bg

        # Extraer canales alpha
        overlay_region = overlay[y1-y:y2-y, x1-x:x2-x]
        alpha = overlay_region[..., 3:] / 255.0
        bg_region = bg[y1:y2, x1:x2]

        # Mezclar imágenes
        bg[y1:y2, x1:x2] = (bg_region * (1 - alpha) + overlay_region[..., :3] * alpha).astype(np.uint8)
        return bg
