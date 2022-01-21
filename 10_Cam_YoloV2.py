import time
import cv2 as cv
import numpy as np
from WebcamVideoStream import WebcamVideoStream
from FaceRecognition import FaceRecognition
from SaveTimings import SaveTimings

# Constants
PATH_CONFIG = '__models/10_YOLOv2/yolov2.cfg'
PATH_WEIGHTS = '__models/10_YOLOv2/yolov2.weights'

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
classifier = cv.dnn.readNetFromDarknet(PATH_CONFIG, PATH_WEIGHTS)


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


def perform_detection(frame: np.ndarray, image_size: int = 416) -> np.ndarray:

    # Resize Image
    frame = cv.resize(frame, (image_size, image_size))
    (H, W) = frame.shape[:2]

    # Output Layers
    ln = classifier.getLayerNames()
    ln = [ln[i - 1] for i in classifier.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(
        frame, 1 / 255.0, (image_size, image_size), swapRB=True, crop=False)
    classifier.setInput(blob)

    # Perform Face Detection
    classifier_output = classifier.forward(ln)

    boxes = []
    confidences = []

    # Filter Detections
    for output in classifier_output:
        for detection in output:
            scores = detection[5:]

            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                # Scale Bbox (Center, W, H) to Image Size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Top Left Corner
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, x + int(width), y + int(height)])
                confidences.append(float(confidence))

    # NMS
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw Detected Faces Into Frame
    draw_bounding_boxes(frame, [boxes[i] for i in idxs])

    return frame


def main() -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings("10_YoloV2")

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
