import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/iesus_robot/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')

sombrero_paths = ["imgs/hats/hat1.png", "imgs/hats/hat2.png", "imgs/hats/hat3.png", "imgs/hats/hat4.png"]
sombrero_imgs = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in sombrero_paths]
current_sombrero_index = 1

def cargar_sombrero(index):
    sombrero_img = sombrero_imgs[index]
    alpha = cv2.split(sombrero_img)[3] / 255.0  # Normalizar el canal alfa
    return sombrero_img, alpha

sombrero_img, alpha = cargar_sombrero(current_sombrero_index)

historial_rostros = []

def agregar_sombrero(frame, sombrero_img, alpha):
    global historial_rostros
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(rostros) > 0:
        historial_rostros.append(rostros)
    else:
        historial_rostros.append([])

    if len(historial_rostros) > 5:
        historial_rostros.pop(0)

    rostros_suavizados = []
    for i in range(len(rostros)):
        x, y, w, h = np.mean([r[i] for r in historial_rostros if len(r) > i], axis=0).astype(int)
        rostros_suavizados.append((x, y, w, h))

    for (x, y, w, h) in rostros_suavizados:
        sombrero_resized = cv2.resize(sombrero_img, (w, int(w * sombrero_img.shape[0] / sombrero_img.shape[1])))
        alpha_resized = cv2.resize(alpha, (w, int(w * alpha.shape[0] / alpha.shape[1])))

        max_sombrero_height = int(h * 0.6)
        if sombrero_resized.shape[0] > max_sombrero_height:
            sombrero_resized = cv2.resize(sombrero_resized, (w, max_sombrero_height))
            alpha_resized = cv2.resize(alpha_resized, (w, max_sombrero_height))
        
        sombrero_x = x
        sombrero_y = y - sombrero_resized.shape[0]
        if sombrero_y < 0:
            sombrero_y = 0

        h, w, _ = sombrero_resized.shape
        sombrero_region = frame[sombrero_y:sombrero_y + h, sombrero_x:sombrero_x + w]
        
        for c in range(0, 3):
            sombrero_region[:, :, c] = (
                alpha_resized * sombrero_resized[:, :, c] + (1 - alpha_resized) * sombrero_region[:, :, c]
            )
        
        frame[sombrero_y:sombrero_y + h, sombrero_x:sombrero_x + w] = sombrero_region

    return frame

def main():
    global current_sombrero_index, sombrero_img, alpha

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame del video.")
            break
        
        frame = agregar_sombrero(frame, sombrero_img, alpha)
        
        cv2.imshow("Filtro de Sombrero", frame)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            current_sombrero_index = (current_sombrero_index + 1) % len(sombrero_imgs)
            sombrero_img, alpha = cargar_sombrero(current_sombrero_index)
        elif key == ord('a'):
            current_sombrero_index = (current_sombrero_index - 1) % len(sombrero_imgs)
            sombrero_img, alpha = cargar_sombrero(current_sombrero_index)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
