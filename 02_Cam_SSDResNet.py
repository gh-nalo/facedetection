import time
import cv2 as cv
import numpy as np
from WebcamVideoStream import WebcamVideoStream
from FaceRecognition import FaceRecognition

# Constants
PATH_PROTO = "./__models/02_SSDResNet/deploy.prototxt.txt"
PATH_MODEL = "./__models/02_SSDResNet/res10_300x300_ssd_iter_140000.caffemodel"

WINDOW_NAME = 'Python Object Detection'

BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 2
TEXT_COLOR = (0, 0, 255)
TEXT_WIDTH = 2

PERFORM_RECOGNITION = True

# Recognition Model
if PERFORM_RECOGNITION:
    recognition = FaceRecognition()

# Detection Model
classifier = cv.dnn.readNetFromCaffe(PATH_PROTO, PATH_MODEL)


def draw_bounding_boxes(frame: np.ndarray, bboxes: np.ndarray, image_size: int) -> None:
    
    for _, isFace, conf, x1, y1, x2, y2, *_ in bboxes:
        
        x1 = int(x1 * image_size)
        y1 = int(y1 * image_size)
        x2 = int(x2 * image_size)
        y2 = int(y2 * image_size)

        if isFace != 1 or conf <= 0.90:
            continue

        if PERFORM_RECOGNITION:

            text = recognition.recognize_face(
                frame, x1, y1, x2, y2
            )

            cv.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                TEXT_COLOR,
                TEXT_WIDTH
            )

        cv.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            BOX_WIDTH,
        )


def perform_detection(frame: np.ndarray, image_size: int = 300) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))

    # Perform Face Detection
    blob = cv.dnn.blobFromImage(frame)
    classifier.setInput(blob)

    bboxes = classifier.forward()

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, bboxes[0][0], image_size)

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Count Frames Per Second
    starting_time = time.time()
    fps_counter = 0

    # Detection Loop
    while True:

        # Capture Frame
        frame_captured, frame = video_capture.read()

        # Stop if There is no Image
        if not frame_captured:
            print("No image returned from camera.")
            break

        # Detect Faces
        frame = perform_detection(frame)

        # Show Results
        cv.imshow(WINDOW_NAME, frame)

        # Count Average Frames Per Second
        fps_counter += 1

        if (time.time() - starting_time) > 1.0:

            print(
                f"FPS: {fps_counter / (time.time() - starting_time):.4f}")

            fps_counter = 0
            starting_time = time.time()

        # Cancel Program With "Q"
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv.destroyAllWindows()
    video_capture.stop()


if __name__ == "__main__":

    main()
