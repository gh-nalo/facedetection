import cv2 as cv
import numpy as np
from FilePaths import Models
from BoxDef import BoxDef


class SSDResNet:
    def __init__(self, recognition=None):
        self.recognition = recognition
        self.classifier = cv.dnn.readNetFromCaffe(
            Models.PATH_SSDRESNET_PROTO,
            Models.PATH_SSDRESNET_WEIGHTS
        )

    def draw_bounding_boxes(self, frame: np.ndarray, bboxes: np.ndarray, image_size: int) -> None:

        for _, isFace, conf, x1, y1, x2, y2, *_ in bboxes:

            x1 = int(x1 * image_size)
            y1 = int(y1 * image_size)
            x2 = int(x2 * image_size)
            y2 = int(y2 * image_size)

            if isFace != 1 or conf <= 0.90:
                continue

            if self.recognition:

                text = self.recognition.recognize_face(
                    frame, x1, y1, x2, y2
                )

                cv.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    BoxDef.TEXT_COLOR,
                    BoxDef.TEXT_WIDTH
                )

            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BoxDef.BOX_COLOR,
                BoxDef.BOX_WIDTH,
            )

    def perform_detection(self, frame: np.ndarray, image_size: int = 300) -> np.ndarray:

        # Resize Image
        frame = cv.resize(frame, (image_size, image_size))

        # Perform Face Detection
        blob = cv.dnn.blobFromImage(frame)
        self.classifier.setInput(blob)

        bboxes = self.classifier.forward()

        # Draw Detected Faces Into Frame
        self.draw_bounding_boxes(frame, bboxes[0][0], image_size)

        return frame
