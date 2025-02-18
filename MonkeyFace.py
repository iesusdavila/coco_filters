import cv2
from utils.AdvancedFaceFilter import AnimalFilter

def main():
    animal_filter = AnimalFilter("monkey")
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
        frame = animal_filter.apply_filter(frame)
        cv2.imshow("Filtro de Animal", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
