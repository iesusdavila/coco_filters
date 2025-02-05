import cv2
import mediapipe as mp
import time
import numpy as np
import os
import glob

# ----------------------------
# Clase base para filtros faciales
# ----------------------------
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

# ----------------------------
# Filtro para gafas con rotación
# ----------------------------
class GlassesFilter(FaceFilter):
    def __init__(self, assets_path):
        super().__init__(assets_path)
        self.required_landmarks = [33, 263, 1]  # Puntos oculares y nariz

    def rotate_image(self, image, angle):
        """Rotación manteniendo transparencia"""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Convertir a coordenadas de píxeles
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            # Calcular ángulo de rotación
            dx = x2 - x1
            dy = y2 - y1
            angle = -np.degrees(np.arctan2(dy, dx))
            
            # Calcular tamaño
            eye_distance = np.hypot(dx, dy)
            glasses_width = int(eye_distance * 1.5)
            glasses_img = self.assets[self.current_asset_idx]
            aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            
            # Procesar imagen
            resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
            rotated = self.rotate_image(resized, angle)
            
            # Posicionamiento
            center_x = (x1 + x2)//2 - rotated.shape[1]//2
            center_y = (y1 + y2)//2 - rotated.shape[0]//2
            
            # Superposición optimizada
            return self.optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

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


# ----------------------------
# Filtro para sombreros
# ----------------------------
class HatFilter(FaceFilter):
    def __init__(self, assets_path):
        super().__init__(assets_path)
        self.required_landmarks = [103, 332]  # Puntos de la frente

    def rotate_image(self, image, angle):
        """Rotación manteniendo transparencia"""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            forehead_left_side = face_landmarks.landmark[103]
            forehead_right_side = face_landmarks.landmark[332]
            
            # Convertir a coordenadas de píxeles
            x1, y1 = int(forehead_left_side.x * w), int(forehead_left_side.y * h)
            x2, y2 = int(forehead_right_side.x * w), int(forehead_right_side.y * h)

            # Calcular ángulo de rotación
            dx = x2 - x1
            dy = y2 - y1
            angle = -np.degrees(np.arctan2(dy, dx))
            
            # Calcular tamaño
            forehead_distance = np.hypot(dx, dy)
            hat_width = int(forehead_distance * 1.8)
            hat_img = self.assets[self.current_asset_idx]
            aspect_ratio = hat_img.shape[0] / hat_img.shape[1]
            hat_height = int(hat_width * aspect_ratio * 0.8)
            
            # Procesar imagen
            resized = cv2.resize(hat_img, (hat_width, hat_height))
            rotated = self.rotate_image(resized, angle)
            
            # Posicionamiento
            center_x = (x1 + x2)//2 - rotated.shape[1]//2
            center_y = (y1 + y2)//2 - rotated.shape[0]//2 - hat_height//3
            
            # Superposición optimizada
            return self.optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

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


# ----------------------------
# Sistema principal
# ----------------------------
class FaceFilterSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))    

        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.active_filters = {
            'glasses': GlassesFilter("imgs/glasses/"),
            'hat': HatFilter("imgs/hats/")
        }
        self.current_filter = 'hat'
        self.prev_time = time.time()
        self.frame_count = 0

    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        self.frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        # Procesar todos los filtros
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for filter_name, filter_obj in self.active_filters.items():
                    frame = filter_obj.apply_filter(frame, face_landmarks, frame.shape[:2])
        
        # Mostrar FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def run(self):
        while self.cap.isOpened():
            frame = self.process_frame()
            if frame is None:
                break
            
            cv2.imshow('Face Filters', frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Navegación entre assets
            if key == ord('d'):
                self.active_filters[self.current_filter].current_asset_idx = (
                    self.active_filters[self.current_filter].current_asset_idx + 1) % len(
                        self.active_filters[self.current_filter].assets)
            elif key == ord('a'):
                self.active_filters[self.current_filter].current_asset_idx = (
                    self.active_filters[self.current_filter].current_asset_idx - 1) % len(
                        self.active_filters[self.current_filter].assets)
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Ejecutar sistema
if __name__ == "__main__":
    system = FaceFilterSystem()
    system.run()