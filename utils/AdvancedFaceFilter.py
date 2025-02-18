import cv2
import mediapipe as mp
import numpy as np

class AnimalFilter:
    def __init__(self, animal_name):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=2)
        self.load_masks(animal_name)
        self.mouth_open_history = []
        self.eyes_open_history = []

    def load_masks(self, animal_name):
        base_path = f"imgs/animals_mask/{animal_name}/"
        self.mask_closed_closed = cv2.imread(base_path + f"{animal_name}_closed_closed.png", cv2.IMREAD_UNCHANGED)
        self.mask_open_closed = cv2.imread(base_path + f"{animal_name}_open_closed.png", cv2.IMREAD_UNCHANGED)
        self.mask_closed_open = cv2.imread(base_path + f"{animal_name}_closed_open.png", cv2.IMREAD_UNCHANGED)
        self.mask_open_open = cv2.imread(base_path + f"{animal_name}_open_open.png", cv2.IMREAD_UNCHANGED)

    def smooth_value(self, history, new_value, max_size=5):
        history.append(new_value)
        if len(history) > max_size:
            history.pop(0)
        return sum(history) / len(history)

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w//2, h//2)

        cos = np.abs(np.cos(np.radians(angle)))
        sin = np.abs(np.sin(np.radians(angle)))
        new_w = int(w * cos + h * sin)
        new_h = int(h * cos + w * sin)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_mat[0, 2] += (new_w - w)//2
        rot_mat[1, 2] += (new_h - h)//2

        rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))
        return rotated

    def optimized_overlay(self, bg, overlay, x, y):
        h_overlay, w_overlay = overlay.shape[:2]
        y1, y2 = max(y, 0), min(y + h_overlay, bg.shape[0])
        x1, x2 = max(x, 0), min(x + w_overlay, bg.shape[1])
        if x1 >= x2 or y1 >= y2:
            return bg
        overlay_region = overlay[y1-y:y2-y, x1-x:x2-x]
        alpha = overlay_region[..., 3:] / 255.0
        bg_region = bg[y1:y2, x1:x2]
        bg[y1:y2, x1:x2] = (bg_region * (1 - alpha) + overlay_region[..., :3] * alpha).astype(np.uint8)
        return bg

    def apply_filter(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
                right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
                nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
                chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
                forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])

                face_width = np.linalg.norm(right_eye - left_eye) * 3.2
                face_height = np.linalg.norm(forehead - chin) * 2
                
                delta_x = right_eye[0] - left_eye[0]
                delta_y = right_eye[1] - left_eye[1]
                angle = -np.degrees(np.arctan2(delta_y, delta_x))
                
                lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
                lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
                mouth_distance = np.linalg.norm(lip_top - lip_bottom)
                mouth_open = self.smooth_value(self.mouth_open_history, mouth_distance > 12) > 0.5
                
                left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
                left_eye_bottom = np.array([landmarks[145].x * w, landmarks[145].y * h])
                right_eye_top = np.array([landmarks[386].x * w, landmarks[386].y * h])
                right_eye_bottom = np.array([landmarks[374].x * w, landmarks[374].y * h])
                eye_open = self.smooth_value(self.eyes_open_history, (np.linalg.norm(left_eye_top - left_eye_bottom) + np.linalg.norm(right_eye_top - right_eye_bottom)) / 2 > 5) > 0.5
                
                mask_to_use = self.mask_open_open if mouth_open and eye_open else self.mask_closed_open if mouth_open else self.mask_open_closed if eye_open else self.mask_closed_closed
                resized_mask = cv2.resize(mask_to_use, (int(face_width), int(face_height)))
                rotated_mask = self.rotate_image(resized_mask, angle)
                top_left_x, top_left_y = int(nose[0] - rotated_mask.shape[1] / 2), int(forehead[1] - rotated_mask.shape[0] * 0.3)
                frame = self.optimized_overlay(frame, rotated_mask, top_left_x, top_left_y)
        return frame
