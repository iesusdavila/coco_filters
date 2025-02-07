import cv2
import mediapipe as mp
import numpy as np

# Inicialización de MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar imágenes del filtro de oso (con canal alfa)
mask_closed_closed = cv2.imread("imgs/animals_mask/bear/bear_closed_closed.png", cv2.IMREAD_UNCHANGED)
mask_open_closed = cv2.imread("imgs/animals_mask/bear/bear_open_closed.png", cv2.IMREAD_UNCHANGED)
mask_closed_open = cv2.imread("imgs/animals_mask/bear/bear_closed_open.png", cv2.IMREAD_UNCHANGED)
mask_open_open = cv2.imread("imgs/animals_mask/bear/bear_open_open.png", cv2.IMREAD_UNCHANGED)

# Historial para suavizar transiciones
mouth_open_history = []
eyes_open_history = []
pos_history = []

# Parámetros para suavizado
ALPHA_SMOOTH = 0.6  # Para suavizar la posición del filtro


def smooth_value(history, new_value, max_size=5):
    """Suaviza cambios en una variable usando un historial de valores recientes."""
    history.append(new_value)
    if len(history) > max_size:
        history.pop(0)
    return sum(history) / len(history)


def smooth_position(history, new_value):
    """Suaviza la posición y tamaño del filtro."""
    if not history:
        history.append(new_value)
    else:
        history.append(ALPHA_SMOOTH * new_value + (1 - ALPHA_SMOOTH) * history[-1])
    if len(history) > 5:
        history.pop(0)
    return history[-1]


def calculate_ear(eye_top, eye_bottom, eye_left, eye_right):
    """Calcula el Eye Aspect Ratio (EAR) para detectar parpadeo."""
    vertical_dist = np.linalg.norm(eye_top - eye_bottom)
    horizontal_dist = np.linalg.norm(eye_left - eye_right)
    return vertical_dist / horizontal_dist


def rotate_image(image, angle):
    """Rota una imagen alrededor de su centro."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated


def optimized_overlay(bg: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Superpone la imagen del filtro con transparencia sobre el fondo."""
    h_overlay, w_overlay = overlay.shape[:2]
    y1, y2 = max(y, 0), min(y + h_overlay, bg.shape[0])
    x1, x2 = max(x, 0), min(x + w_overlay, bg.shape[1])
    
    if x1 >= x2 or y1 >= y2:
        return bg

    overlay_region = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = overlay_region[..., 3:] / 255.0
    bg_region = bg[y1:y2, x1:x2]

    bg[y1:y2, x1:x2] = (bg_region * (1 - alpha) + overlay_region[..., :3] * alpha).astype(np.uint8)
    return bg


GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Puntos clave del rostro
            left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
            right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
            nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
            chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
            forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])

            # Tamaño del rostro
            face_width = smooth_position(pos_history, np.linalg.norm(right_eye - left_eye) * 3.2)
            face_height = smooth_position(pos_history, np.linalg.norm(forehead - chin) * 2)

            # Ángulo de rotación
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            angle = smooth_position(pos_history, -np.degrees(np.arctan2(delta_y, delta_x)))

            # Detección de boca abierta
            lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
            lip_bottom = np.array([landmarks[14].x * w, landmarks[14].y * h])
            mouth_distance = np.linalg.norm(lip_top - lip_bottom)
            mouth_open = smooth_value(mouth_open_history, mouth_distance > 12) > 0.5

            # Detección de ojos abiertos con EAR
            left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
            left_eye_bottom = np.array([landmarks[145].x * w, landmarks[145].y * h])
            left_eye_left = np.array([landmarks[33].x * w, landmarks[33].y * h])
            left_eye_right = np.array([landmarks[133].x * w, landmarks[133].y * h])

            right_eye_top = np.array([landmarks[386].x * w, landmarks[386].y * h])
            right_eye_bottom = np.array([landmarks[374].x * w, landmarks[374].y * h])
            right_eye_left = np.array([landmarks[362].x * w, landmarks[362].y * h])
            right_eye_right = np.array([landmarks[263].x * w, landmarks[263].y * h])

            ear_left = calculate_ear(left_eye_top, left_eye_bottom, left_eye_left, left_eye_right)
            ear_right = calculate_ear(right_eye_top, right_eye_bottom, right_eye_left, right_eye_right)
            eye_open = smooth_value(eyes_open_history, (ear_left + ear_right) / 2 > 0.25) > 0.5

            # Selección de filtro según expresión facial
            mask_to_use = mask_open_open if mouth_open and eye_open else \
                          mask_closed_open if mouth_open else \
                          mask_open_closed if eye_open else mask_closed_closed

            resized_mask = cv2.resize(mask_to_use, (int(face_width), int(face_height)))
            rotated_mask = rotate_image(resized_mask, angle)

            frame = optimized_overlay(frame, rotated_mask, int(nose[0] - face_width // 2), int(forehead[1] - face_height * 0.3))

    cv2.imshow("Filtro de Oso Mejorado", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
