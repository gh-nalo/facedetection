import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from WebcamVideoStream import WebcamVideoStream
from FaceRecognition import FaceRecognition
from SaveTimings import SaveTimings

# Constants
PATH_MODEL = '__models/06_RFCN_resnet101/rfcn_graph.pb'

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
MIN_CONFIDENCE = 0.8
DETECTION_GRAPH = tf.Graph()
SESSION = 0

with DETECTION_GRAPH.as_default():
    od_graph_def = tf.compat.v1.GraphDef()  # CHANGED -- instead of tf.GraphDef()

    with tf.compat.v2.io.gfile.GFile(PATH_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    SESSION = tf.compat.v1.Session(graph=DETECTION_GRAPH)


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


def perform_detection(frame: np.ndarray, image_size: int = 640) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))

    # Expand Image Dimensions For Model ([1, None, None, 3])
    image_np_expanded = np.expand_dims(frame, axis=0)

    # Image
    image_tensor = DETECTION_GRAPH.get_tensor_by_name('image_tensor:0')

    # Perform Face Detection
    (boxes, scores, _, _) = SESSION.run(
        [
            DETECTION_GRAPH.get_tensor_by_name(
                'detection_boxes:0'),  # Object Bounding Box
            DETECTION_GRAPH.get_tensor_by_name(
                'detection_scores:0'),  # Confidence
            DETECTION_GRAPH.get_tensor_by_name(
                'detection_classes:0'),  # Id Of Detected Class
            DETECTION_GRAPH.get_tensor_by_name(
                'num_detections:0')  # Number Of Detections
        ],
        feed_dict={
            image_tensor: image_np_expanded
        }
    )

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    im_height, im_width, _ = frame.shape

    bboxes = []

    for i in range(boxes.shape[0]):
        if scores[i] >= MIN_CONFIDENCE:
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            x, y, w, h = int(left), int(top), int(
                right - left), int(bottom - top)

            bboxes.append([x, y, x + w, y + h])

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, bboxes)

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings("06_RFCN")

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
