import cv2
import numpy as np

class Filtro:
    def __init__(self, cascade_path, imagenes):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.imagenes = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in imagenes]
        self.indice_actual = 0
        self.historial_rostros = []
    
    def cambiar_filtro(self, direccion):
        """Cambia el filtro según la dirección (1: siguiente, -1: anterior)."""
        self.indice_actual = (self.indice_actual + direccion) % len(self.imagenes)
    
    def aplicar_filtro(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(rostros) > 0:
            self.historial_rostros.append(rostros)
        else:
            self.historial_rostros.append([])

        if len(self.historial_rostros) > 5:
            self.historial_rostros.pop(0)

        rostros_suavizados = [
            tuple(np.mean([r[i] for r in self.historial_rostros if len(r) > i], axis=0).astype(int))
            for i in range(len(rostros))
        ]

        for (x, y, w, h) in rostros_suavizados:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.agregar_sobre(frame, x, y, w, h)

        return frame

    def agregar_sobre(self, frame, x, y, w, h):
        """Este método debe ser implementado por cada tipo de filtro."""
        raise NotImplementedError


class SombreroFiltro(Filtro):
    def agregar_sobre(self, frame, x, y, w, h):
        imagen_sombrero = self.imagenes[self.indice_actual]
        
        if imagen_sombrero is None:
            return

        sombrero_ancho = w
        sombrero_alto = int(w * imagen_sombrero.shape[0] / imagen_sombrero.shape[1])
        sombrero_y = max(y - sombrero_alto, 0)

        sombrero_redimensionado = cv2.resize(imagen_sombrero, (sombrero_ancho, sombrero_alto))
        
        max_sombrero_height = int(h * 0.5)
        if sombrero_redimensionado.shape[0] > max_sombrero_height:
            sombrero_redimensionado = cv2.resize(sombrero_redimensionado, (w, max_sombrero_height))

        sombrero_y = y - sombrero_redimensionado.shape[0]
        if sombrero_y < 0:
            sombrero_y = 0

        self.overlay_transparente(frame, sombrero_redimensionado, x, sombrero_y)

    def overlay_transparente(self, fondo, overlay, x, y):
        h, w, _ = overlay.shape
        bh, bw, _ = fondo.shape

        if x + w > bw or y + h > bh:
            return

        alpha_overlay = overlay[:, :, 3] / 255.0
        for c in range(3):
            fondo[y:y+h, x:x+w, c] = (
                alpha_overlay * overlay[:, :, c] + (1 - alpha_overlay) * fondo[y:y+h, x:x+w, c]
            )


def main(filtros):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for filtro in filtros:
            frame = filtro.aplicar_filtro(frame)

        cv2.imshow("Filtros", frame)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            for filtro in filtros:
                filtro.cambiar_filtro(1)
        elif key == ord('a'):
            for filtro in filtros:
                filtro.cambiar_filtro(-1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ruta_cascade = "/home/iesus_robot/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    imagenes_sombrero = ["imgs/hats/hat1.png", "imgs/hats/hat2.png", "imgs/hats/hat3.png", 
        "imgs/hats/hat4.png", "imgs/hats/hat5.png", "imgs/hats/hat6.png", 
        "imgs/hats/hat7.png", "imgs/hats/hat8.png", "imgs/hats/hat9.png",
        "imgs/hats/hat10.png", "imgs/hats/hat11.png", "imgs/hats/hat12.png",
        "imgs/hats/hat13.png", "imgs/hats/hat14.png", "imgs/hats/hat15.png",
        "imgs/hats/hat16.png", "imgs/hats/hat17.png", "imgs/hats/hat18.png",
        "imgs/hats/hat19.png", "imgs/hats/hat20.png", "imgs/hats/hat21.png",
        "imgs/hats/hat22.png", "imgs/hats/hat23.png", "imgs/hats/hat24.png",
        "imgs/hats/hat25.png", "imgs/hats/hat26.png"]
    
    sombrero_filtro = SombreroFiltro(ruta_cascade, imagenes_sombrero)
    main([sombrero_filtro])
