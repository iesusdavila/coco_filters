import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mask_closed_closed = cv2.imread("imgs/bear/bear_closed_closed.png", cv2.IMREAD_UNCHANGED)  # Eyes close, mouth close
mask_open_closed = cv2.imread("imgs/bear/bear_open_closed.png", cv2.IMREAD_UNCHANGED)  # Eyes open, mouth close
mask_closed_open = cv2.imread("imgs/bear/bear_closed_open.png", cv2.IMREAD_UNCHANGED)  # Eyes close, mouth open
mask_open_open = cv2.imread("imgs/bear/bear_open_open.png", cv2.IMREAD_UNCHANGED)  # Eyes open, mouth open

mouth_open_history = []
eyes_open_history = []

def smooth_value(history, new_value, max_size=5):
    history.append(new_value)
    if len(history) > max_size:
        history.pop(0)
    return sum(history) / len(history)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    new_w = int(w * cos_angle + h * sin_angle)
    new_h = int(h * cos_angle + w * sin_angle)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def overlay_image(bg, fg, position):
    x, y = position
    h, w, _ = fg.shape

    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg

    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(0, 3):
        bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y+h, x:x+w, c])

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
            mouth_open = smooth_value(mouth_open_history, mouth_distance > 12) > 0.5

            left_eye_top = np.array([landmarks[159].x * w, landmarks[159].y * h])
            left_eye_bottom = np.array([landmarks[145].x * w, landmarks[145].y * h])
            right_eye_top = np.array([landmarks[386].x * w, landmarks[386].y * h])
            right_eye_bottom = np.array([landmarks[374].x * w, landmarks[374].y * h])

            left_eye_distance = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_distance = np.linalg.norm(right_eye_top - right_eye_bottom)
            eye_open = smooth_value(eyes_open_history, (left_eye_distance + right_eye_distance) / 2 > 5) > 0.5

            if mouth_open and eye_open:
                mask_to_use = mask_open_open
            elif mouth_open and not eye_open:
                mask_to_use = mask_closed_open
            elif not mouth_open and eye_open:
                mask_to_use = mask_open_closed
            else:
                mask_to_use = mask_closed_closed

            resized_mask = cv2.resize(mask_to_use, (int(face_width), int(face_height)))

            rotated_mask = rotate_image(resized_mask, angle)

            new_h, new_w, _ = rotated_mask.shape
            top_left = (int(nose[0] - new_w / 2), int(forehead[1] - new_h * 0.3))

            frame = overlay_image(frame, rotated_mask, top_left)

    cv2.imshow("Filtro de Oso con Parpadeo y Boca Animada", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
