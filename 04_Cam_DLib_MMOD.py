import time
import cv2 as cv
import numpy as np
import dlib
from WebcamVideoStream import WebcamVideoStream
from FaceRecognition import FaceRecognition
from SaveTimings import SaveTimings

# Constants
PATH_MODEL = "./__models/04_MMOD/mmod_human_face_detector.dat"

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
classifier = dlib.cnn_face_detection_model_v1(PATH_MODEL)


def convert_rect_to_bb(rect, orig_height, orig_width, input_size):
    # extract the starting and ending (x, y)-coordinates of the bounding box
    x1 = int(rect.left() * (orig_width / input_size))
    y1 = int(rect.top() * (orig_height / input_size))
    x2 = int(rect.right() * (orig_width / input_size))
    y2 = int(rect.bottom() * (orig_height / input_size))

    return (x1, y1, x2, y2)


def draw_bounding_boxes(frame: np.ndarray, bboxes: np.ndarray) -> None:

    for x1, y1, x2, y2 in bboxes:

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


def perform_detection(frame: np.ndarray, image_size: int = 600) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))

    # Perform Face Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rects = classifier(rgb, 0)
    bboxes = [convert_rect_to_bb(
        r.rect, frame.shape[0], frame.shape[1], image_size) for r in rects
    ]

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, bboxes)

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings("04_DLib_MMOD")

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
