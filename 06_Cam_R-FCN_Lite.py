from FaceRecognition import FaceRecognition
from WebcamVideoStream import WebcamVideoStream
from SaveTimings import SaveTimings
import tensorflow as tf
import time
import cv2 as cv
import numpy as np

# Constants
PATH_MODEL = "./__models/06_RFCN_resnet101/rfcn_640.tflite"

WINDOW_NAME = 'Python Object Detection'

BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 2
TEXT_COLOR = (0, 0, 255)
TEXT_WIDTH = 2

PERFORM_RECOGNITION = True

# Recognition Model
if PERFORM_RECOGNITION:
    recognition = FaceRecognition()

# Load the TFLite model and allocate tensors.
MIN_CONFIDENCE = 0.5

interpreter = tf.lite.Interpreter(PATH_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


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


def perform_detection(frame: np.ndarray, image_size: int = 640) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))
    frame = cv.resize(frame, (width, height))

    input_data = np.expand_dims(frame, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 175.5) / 175.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform Face Detection
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    im_height, im_width, _ = frame.shape

    bboxes = []

    for i in range(boxes.shape[0]):
        if scores[i] >= MIN_CONFIDENCE:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            x1, y1, x2, y2 = int(left), int(top), int(
                right - left), int(bottom - top)

            bboxes.append([x1, y1, x2, y2])

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, bboxes)

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings("06_RFCN_Lite")

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
