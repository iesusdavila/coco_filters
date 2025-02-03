import cv2
import numpy as np

class Filter:
    def __init__(self, cascade_path, images):
        """
        Initialize the Filter class with a face cascade and a list of images.

        :param cascade_path: Path to the Haar cascade file for face detection.
        :param images: List of image file paths to be used as filters.
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in images]
        self.current_index = 0
        self.face_history = []

    def change_filter(self, direction):
        """
        Change the current filter based on the direction.

        :param direction: Direction to change the filter (1 for next, -1 for previous).
        """
        self.current_index = (self.current_index + direction) % len(self.images)

    def apply_filter(self, frame):
        """
        Apply the current filter to the given frame.

        :param frame: The video frame to which the filter will be applied.
        :return: The frame with the filter applied.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            self.face_history.append(faces)
        else:
            self.face_history.append([])

        if len(self.face_history) > 5:
            self.face_history.pop(0)

        smooth_faces = [
            tuple(np.mean([r[i] for r in self.face_history if len(r) > i], axis=0).astype(int))
            for i in range(len(faces))
        ]

        for (x, y, w, h) in smooth_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.add_about(frame, x, y, w, h)

        return frame

    def add_about(self, frame, x, y, w, h):
        """
        Abstract method to add specific filter details to the frame.

        :param frame: The video frame to which the filter details will be added.
        :param x: The x-coordinate of the detected face.
        :param y: The y-coordinate of the detected face.
        :param w: The width of the detected face.
        :param h: The height of the detected face.
        """
        raise NotImplementedError


class HatFilter(Filter):
    def add_about(self, frame, x, y, w, h):
        """
        Add a hat filter to the detected face in the frame.

        :param frame: The video frame to which the hat filter will be added.
        :param x: The x-coordinate of the detected face.
        :param y: The y-coordinate of the detected face.
        :param w: The width of the detected face.
        :param h: The height of the detected face.
        """
        hat_image = self.images[self.current_index]
        
        if hat_image is None:
            return

        hat_width = w
        hat_height = int(w * hat_image.shape[0] / hat_image.shape[1])
        hat_y = max(y - hat_height, 0)

        resized_hat = cv2.resize(hat_image, (hat_width, hat_height))
        
        max_hat_height = int(h * 0.5)
        if resized_hat.shape[0] > max_hat_height:
            resized_hat = cv2.resize(resized_hat, (w, max_hat_height))

        hat_y = y - resized_hat.shape[0]
        if hat_y < 0:
            hat_y = 0

        self.overlay_transparent(frame, resized_hat, x, hat_y)

    def overlay_transparent(self, background, overlay, x, y):
        """
        Overlay a transparent image on the background frame.

        :param background: The background frame.
        :param overlay: The transparent image to overlay.
        :param x: The x-coordinate where the overlay will be placed.
        :param y: The y-coordinate where the overlay will be placed.
        """
        h, w, _ = overlay.shape
        bh, bw, _ = background.shape

        if x + w > bw or y + h > bh:
            return

        alpha_overlay = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha_overlay * overlay[:, :, c] + (1 - alpha_overlay) * background[y:y+h, x:x+w, c]
            )


def main(filters):
    """
    Main function to capture video and apply filters.

    :param filters: List of filter objects to be applied to the video frames.
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for filter in filters:
            frame = filter.apply_filter(frame)

        cv2.imshow("filters", frame)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            for filter in filters:
                filter.change_filter(1)
        elif key == ord('a'):
            for filter in filters:
                filter.change_filter(-1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cascade_route = "/home/iesus_robot/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    images_hat = ["imgs/hats/hat1.png", "imgs/hats/hat2.png", "imgs/hats/hat3.png", 
        "imgs/hats/hat4.png", "imgs/hats/hat5.png", "imgs/hats/hat6.png", 
        "imgs/hats/hat7.png", "imgs/hats/hat8.png", "imgs/hats/hat9.png",
        "imgs/hats/hat10.png", "imgs/hats/hat11.png", "imgs/hats/hat12.png",
        "imgs/hats/hat13.png", "imgs/hats/hat14.png", "imgs/hats/hat15.png",
        "imgs/hats/hat16.png", "imgs/hats/hat17.png", "imgs/hats/hat18.png",
        "imgs/hats/hat19.png", "imgs/hats/hat20.png", "imgs/hats/hat21.png",
        "imgs/hats/hat22.png", "imgs/hats/hat23.png", "imgs/hats/hat24.png",
        "imgs/hats/hat25.png", "imgs/hats/hat26.png"]
    
    filter_hat = HatFilter(cascade_route, images_hat)
    main([filter_hat])
