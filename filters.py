import cv2
import numpy as np

class SombreroFiltro:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('/home/iesus_robot/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
        self.sombrero_paths = ["imgs/hats/hat1.png", "imgs/hats/hat2.png", "imgs/hats/hat3.png", 
                                "imgs/hats/hat4.png", "imgs/hats/hat5.png", "imgs/hats/hat6.png", 
                                "imgs/hats/hat7.png", "imgs/hats/hat8.png", "imgs/hats/hat9.png",
                                "imgs/hats/hat10.png", "imgs/hats/hat11.png", "imgs/hats/hat12.png",
                                "imgs/hats/hat13.png", "imgs/hats/hat14.png", "imgs/hats/hat15.png",
                                "imgs/hats/hat16.png", "imgs/hats/hat17.png", "imgs/hats/hat18.png",
                                "imgs/hats/hat19.png", "imgs/hats/hat20.png", "imgs/hats/hat21.png",
                                "imgs/hats/hat22.png", "imgs/hats/hat23.png", "imgs/hats/hat24.png",
                                "imgs/hats/hat25.png", "imgs/hats/hat26.png"]
        self.sombrero_imgs = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in self.sombrero_paths]
        self.current_sombrero_index = 1
        self.sombrero_img, self.alpha = self.cargar_sombrero(self.current_sombrero_index)
        self.historial_rostros = []

    def cargar_sombrero(self, index):
        sombrero_img = self.sombrero_imgs[index]
        alpha = cv2.split(sombrero_img)[3] / 255.0  # Normalizar el canal alfa
        return sombrero_img, alpha

    def agregar_sombrero(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(rostros) > 0:
            self.historial_rostros.append(rostros)
        else:
            self.historial_rostros.append([])

        if len(self.historial_rostros) > 5:
            self.historial_rostros.pop(0)

        rostros_suavizados = []
        for i in range(len(rostros)):
            x, y, w, h = np.mean([r[i] for r in self.historial_rostros if len(r) > i], axis=0).astype(int)
            rostros_suavizados.append((x, y, w, h))

        for (x, y, w, h) in rostros_suavizados:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            sombrero_resized = cv2.resize(self.sombrero_img, (w, int(w * self.sombrero_img.shape[0] / self.sombrero_img.shape[1])))
            alpha_resized = cv2.resize(self.alpha, (w, int(w * self.alpha.shape[0] / self.alpha.shape[1])))

            max_sombrero_height = int(h * 0.5)
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

def main(filtros):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame del video.")
            break
        
        for filtro in filtros:
            frame = filtro.agregar_sombrero(frame)
        
        cv2.imshow("Filtros", frame)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            for filtro in filtros:
                filtro.current_sombrero_index = (filtro.current_sombrero_index + 1) % len(filtro.sombrero_imgs)
                filtro.sombrero_img, filtro.alpha = filtro.cargar_sombrero(filtro.current_sombrero_index)
        elif key == ord('a'):
            for filtro in filtros:
                filtro.current_sombrero_index = (filtro.current_sombrero_index - 1) % len(filtro.sombrero_imgs)
                filtro.sombrero_img, filtro.alpha = filtro.cargar_sombrero(filtro.current_sombrero_index)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sombrero_filtro = SombreroFiltro()
    # Aquí se pueden agregar más filtros en la lista
    main([sombrero_filtro])
