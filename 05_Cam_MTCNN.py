import time
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from WebcamVideoStream import WebcamVideoStream
from FaceRecognition import FaceRecognition
from SaveTimings import SaveTimings

# Constants
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
classifier = MTCNN()


def draw_bounding_boxes(frame: np.ndarray, bboxes: np.ndarray) -> None:

    for x1, y1, width, height in bboxes:

        if PERFORM_RECOGNITION:

            text = recognition.recognize_face(
                frame, x1, y1, x1 + width, y1 + height
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
            (x1 + width, y1 + height),
            BOX_COLOR,
            BOX_WIDTH,
        )


def perform_detection(frame: np.ndarray, image_size: int = 600) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))

    # Perform Face Detection
    faces = classifier.detect_faces(frame)
    bboxes = [face['box'] for face in faces]

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, bboxes)

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings("05_MTCNN")

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

            fps = fps_counter / (time.time() - starting_time)

            if timer.new_value(fps) > 200:
                timer.save_results()
                break

            print(f"FPS: {fps:.4f}")

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
