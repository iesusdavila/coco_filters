import mediapipe as mp
import time
import cv2
from utils.Filters import GlassesFilter, HatFilter, NoseFilter, MouthFilter, FaceMaskFilter

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