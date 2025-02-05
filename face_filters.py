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
            return super().optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

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
            return super().optimized_overlay(frame, rotated, center_x, center_y)
        
        except (IndexError, cv2.error):
            return frame

# ----------------------------
# Filtro para nariz
# ----------------------------
class NoseFilter(FaceFilter):
    def __init__(self, assets_path):
        super().__init__(assets_path)
        self.required_landmarks = [1, 4, 5, 6, 168]  # Puntos clave de la nariz

    def rotate_image(self, image, angle):
        """Rotación optimizada con padding inteligente"""
        if angle == 0:
            return image

        h, w = image.shape[:2]
        center = (w//2, h//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular nuevo tamaño con mínima pérdida
        cos = np.abs(rot_mat[0,0])
        sin = np.abs(rot_mat[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0,2] += (new_w - w)//2
        rot_mat[1,2] += (new_h - h)//2
        
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                            flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        if not self.assets or not face_landmarks:
            return frame
        
        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark
            
            # Puntos de referencia principales
            nose_tip = landmarks[4]
            nose_bridge = landmarks[6]
            left_nostril = landmarks[98]
            right_nostril = landmarks[327]

            # Convertir a coordenadas de píxeles
            x_tip, y_tip = int(nose_tip.x * w), int(nose_tip.y * h)
            x_bridge, y_bridge = int(nose_bridge.x * w), int(nose_bridge.y * h)
            x_left, y_left = int(left_nostril.x * w), int(left_nostril.y * h)
            x_right, y_right = int(right_nostril.x * w), int(right_nostril.y * h)

            # Validar visibilidad de puntos
            if not all([self._is_visible(lm, w, h) for lm in [nose_tip, nose_bridge]]):
                return frame

            # Calcular tamaño y orientación
            nose_width = abs(x_right - x_left)
            vertical_distance = abs(y_bridge - y_tip)
            nose_height = int(vertical_distance * 1.2)
            
            # Calcular ángulo de rotación basado en la inclinación de la cabeza
            dx = x_right - x_left
            dy = y_right - y_left
            angle = -np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0

            # Obtener asset actual
            nose_asset = self.assets[self.current_asset_idx]
            
            # Calcular proporciones manteniendo aspecto
            target_width = int(nose_width * 1.5)
            aspect_ratio = nose_asset.shape[0] / nose_asset.shape[1]
            target_height = int(target_width * aspect_ratio)
            
            # Procesar imagen
            resized = cv2.resize(nose_asset, (target_width, target_height))
            rotated = self.rotate_image(resized, angle)

            # Posicionamiento centrado en la punta de la nariz
            pos_x = x_tip - rotated.shape[1] // 2
            pos_y = y_tip - rotated.shape[0] // 2

            # Validar posición dentro del frame
            if not self._is_valid_position(pos_x, pos_y, rotated.shape, w, h):
                return frame

            return super().optimized_overlay(frame, rotated, pos_x, pos_y)

        except (IndexError, cv2.error) as e:
            print(f"Error aplicando filtro de nariz: {e}")
            return frame

    def _is_visible(self, landmark, img_w, img_h):
        """Valida si un landmark está dentro del área visible"""
        return (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1 and
                img_w * 0.1 <= landmark.x * img_w <= img_w * 0.9 and
                img_h * 0.1 <= landmark.y * img_h <= img_h * 0.9)

    def _is_valid_position(self, x, y, shape, img_w, img_h):
        """Valida que la posición no esté fuera de los límites"""
        h_asset, w_asset = shape[:2]
        return (x >= -w_asset * 0.2 and 
                y >= -h_asset * 0.2 and
                x + w_asset <= img_w * 1.2 and
                y + h_asset <= img_h * 1.2)

# ----------------------------
# Filtro para boca
# ----------------------------
class MouthFilter(FaceFilter):
    def __init__(self, assets_path):
        super().__init__(assets_path)
        self.required_landmarks = [
            61, 291,  # Esquinas de la boca
            13, 14,   # Línea media labial
            78, 308,  # Contorno superior labio
            37, 267,  # Contorno inferior labio
        ]

    def rotate_image(self, image, angle):
        """Rotación manteniendo transparencia"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0, 2] += (new_w - w) // 2
        rot_mat[1, 2] += (new_h - h) // 2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def apply_filter(self, frame, face_landmarks, frame_size):
        if not self.assets or not face_landmarks:
            return frame

        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark

            # Puntos clave de la boca
            left_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
            right_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
            upper_lip = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            lower_lip = (int(landmarks[17].x * w), int(landmarks[17].y * h))

            # Calcular tamaño y posición de la boca
            mouth_width = abs(right_corner[0] - left_corner[0])
            mouth_height = abs(lower_lip[1] - upper_lip[1])

            if mouth_width == 0 or mouth_height == 0:
                return frame

            # Cargar el asset actual
            mouth_asset = self.assets[self.current_asset_idx]

            # Redimensionar el asset según el tamaño de la boca
            aspect_ratio = mouth_asset.shape[0] / mouth_asset.shape[1]
            target_width = int(mouth_width)
            target_height = int(target_width * aspect_ratio)
            resized_asset = cv2.resize(mouth_asset, (target_width, target_height))

            # Calcular ángulo de rotación basado en la inclinación de la boca
            dx = right_corner[0] - left_corner[0]
            dy = right_corner[1] - left_corner[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            # Rotar el asset
            rotated_asset = self.rotate_image(resized_asset, angle)

            # Posicionamiento centrado en la boca
            center_x = (left_corner[0] + right_corner[0]) // 2 - rotated_asset.shape[1] // 2
            center_y = (upper_lip[1] + lower_lip[1]) // 2 - rotated_asset.shape[0] // 2

            # Validar posición dentro del frame
            if not self._validate_position(center_x, center_y, rotated_asset.shape, w, h):
                return frame

            # Superposición del filtro en el frame
            return super().optimized_overlay(frame, rotated_asset, center_x, center_y)
            
        except (IndexError, cv2.error) as e:
            print(f"Error en filtro bucal: {str(e)}")
            return frame

    def _validate_position(self, x, y, shape, img_w, img_h):
        """Valida que el filtro esté dentro de los límites faciales razonables"""
        h_asset, w_asset = shape[:2]
        return (
            x > -w_asset * 0.2 and
            y > -h_asset * 0.2 and
            x + w_asset < img_w * 1.2 and
            y + h_asset < img_h * 1.2 and
            w_asset > 10 and  # Tamaño mínimo
            h_asset > 10
        )

# ----------------------------
# Filtro para mascaras
# ----------------------------
class FaceMaskFilter(FaceFilter):
    def __init__(self, assets_path):
        super().__init__(assets_path)
        self.required_landmarks = [
            10,  # Punto superior de la frente
            152,  # Punto inferior del mentón
            234,  # Lado izquierdo de la cara
            454,  # Lado derecho de la cara
            61, 291,  # Esquinas de la boca
            1,  # Nariz
            5,  # Punta de la nariz
            2,  # Ceja derecha
            5,  # Ceja izquierda
        ]

    def apply_filter(self, frame, face_landmarks, frame_size):
        if not self.assets or not face_landmarks:
            return frame

        try:
            h, w = frame_size
            landmarks = face_landmarks.landmark

            # Puntos clave de la cara
            top_forehead = (int(landmarks[10].x * w), int(landmarks[10].y * h))
            chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            left_side = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            right_side = (int(landmarks[454].x * w), int(landmarks[454].y * h))

            # Calcular el tamaño y posición de la máscara
            face_width = abs(right_side[0] - left_side[0])
            face_height = abs(chin[1] - top_forehead[1])

            if face_width == 0 or face_height == 0:
                return frame

            # Cargar el asset actual
            mask_asset = self.assets[self.current_asset_idx]

            # Redimensionar el asset según el tamaño de la cara
            aspect_ratio = mask_asset.shape[0] / mask_asset.shape[1]
            target_width = int(face_width * 2.0)
            target_height = int(target_width * aspect_ratio)
            resized_asset = cv2.resize(mask_asset, (target_width, target_height))

            # Calcular ángulo de rotación basado en la inclinación de la cara
            dx = right_side[0] - left_side[0]
            dy = right_side[1] - left_side[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            # Rotar el asset
            rotated_asset = self.rotate_image(resized_asset, angle)

            # Posicionamiento centrado en la cara
            center_x = (left_side[0] + right_side[0]) // 2 - rotated_asset.shape[1] // 2
            center_y = (top_forehead[1] + chin[1]) // 2 - rotated_asset.shape[0] // 2

            # Validar posición dentro del frame
            if not self._validate_position(center_x, center_y, rotated_asset.shape, w, h):
                return frame

            # Superposición del filtro en el frame
            return super().optimized_overlay(frame, rotated_asset, center_x, center_y)

        except (IndexError, cv2.error) as e:
            print(f"Error en filtro de máscara facial: {str(e)}")
            return frame

    def rotate_image(self, image, angle):
        """Rotación manteniendo transparencia"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rot_mat[0, 2] += (new_w - w) // 2
        rot_mat[1, 2] += (new_h - h) // 2
        return cv2.warpAffine(image, rot_mat, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def _validate_position(self, x, y, shape, img_w, img_h):
        """Valida que el filtro esté dentro de los límites faciales razonables"""
        h_asset, w_asset = shape[:2]
        return (
            x > -w_asset * 0.2 and
            y > -h_asset * 0.2 and
            x + w_asset < img_w * 1.2 and
            y + h_asset < img_h * 1.2 and
            w_asset > 10 and  # Tamaño mínimo
            h_asset > 10
        )

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
            'hat': HatFilter("imgs/hats/"),
            'nose': NoseFilter("imgs/noses/"),
            'mouth': MouthFilter("imgs/mouths/"),
        }
        self.is_face_selected = False
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
            if key == ord('e') and not self.is_face_selected:
                self.active_filters['hat'].current_asset_idx = (
                    self.active_filters['hat'].current_asset_idx + 1) % len(
                        self.active_filters['hat'].assets)
            elif key == ord('q') and not self.is_face_selected:
                self.active_filters['hat'].current_asset_idx = (
                    self.active_filters['hat'].current_asset_idx - 1) % len(
                        self.active_filters['hat'].assets)
            elif key == ord('d') and not self.is_face_selected:
                self.active_filters['glasses'].current_asset_idx = (
                    self.active_filters['glasses'].current_asset_idx + 1) % len(
                        self.active_filters['glasses'].assets)
            elif key == ord('a') and not self.is_face_selected:
                self.active_filters['glasses'].current_asset_idx = (
                    self.active_filters['glasses'].current_asset_idx - 1) % len(
                        self.active_filters['glasses'].assets)
            elif key == ord('w') and not self.is_face_selected:
                self.active_filters['nose'].current_asset_idx = (
                    self.active_filters['nose'].current_asset_idx + 1) % len(
                        self.active_filters['nose'].assets)
            elif key == ord('s') and not self.is_face_selected:
                self.active_filters['nose'].current_asset_idx = (
                    self.active_filters['nose'].current_asset_idx - 1) % len(
                        self.active_filters['nose'].assets)
            elif key == ord('c') and not self.is_face_selected:
                self.active_filters['mouth'].current_asset_idx = (
                    self.active_filters['mouth'].current_asset_idx + 1) % len(
                        self.active_filters['mouth'].assets)
            elif key == ord('z') and not self.is_face_selected:
                self.active_filters['mouth'].current_asset_idx = (
                    self.active_filters['mouth'].current_asset_idx - 1) % len(
                        self.active_filters['mouth'].assets)
            elif key == ord('f'):
                self.is_face_selected = not self.is_face_selected
                if self.is_face_selected:
                    self.active_filters = {
                        'face': FaceMaskFilter("imgs/faces/")
                    }
                else:
                    self.active_filters = {
                        'glasses': GlassesFilter("imgs/glasses/"),
                        'hat': HatFilter("imgs/hats/"),
                        'nose': NoseFilter("imgs/noses/"),
                        'mouth': MouthFilter("imgs/mouths/"),
                    }
            elif key == ord('m') and self.is_face_selected:
                self.active_filters['face'].current_asset_idx = (
                    self.active_filters['face'].current_asset_idx + 1) % len(
                        self.active_filters['face'].assets)
            elif key == ord('n') and self.is_face_selected:
                self.active_filters['face'].current_asset_idx = (
                    self.active_filters['face'].current_asset_idx - 1) % len(
                        self.active_filters['face'].assets)
            elif key == ord('l'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Ejecutar sistema
if __name__ == "__main__":
    system = FaceFilterSystem()
    system.run()