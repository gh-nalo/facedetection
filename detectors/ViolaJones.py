import cv2 as cv
import numpy as np
from FilePaths import Models
from BoxDef import BoxDef


class ViolaJones:

    def __init__(self, recognition=None):
        self.recognition = recognition
        self.classifier = cv.CascadeClassifier(Models.PATH_VIOLAJONES)

    def draw_bounding_boxes(self, frame: np.ndarray, bboxes: list) -> None:
        for x1, y1, width, height in bboxes:

            if self.recognition:

                text = self.recognition.recognize_face(
                    frame, x1, y1, x1 + width, y1 + height
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
                (x1 + width, y1 + height),
                BoxDef.BOX_COLOR,
                BoxDef.BOX_WIDTH,
            )

    def perform_detection(self, frame: np.ndarray, image_size: int = 640) -> np.ndarray:

        # Resize Image
        frame = cv.resize(frame, (image_size, image_size))

        # Perform Face Detection
        bboxes = self.classifier.detectMultiScale(
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        )

        self.draw_bounding_boxes(frame, bboxes)

        return frame
