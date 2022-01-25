import cv2 as cv
import numpy as np
import dlib
from FilePaths import Models
from BoxDef import BoxDef


class DLibMMOD:
    def __init__(self, recognition=None):
        self.recognition = recognition
        self.classifier = dlib.cnn_face_detection_model_v1(Models.PATH_MMOD)

    def convert_rect_to_bb(self, rect, orig_height, orig_width, input_size):
        # extract the starting and ending (x, y)-coordinates of the bounding box
        x1 = int(rect.left() * (orig_width / input_size))
        y1 = int(rect.top() * (orig_height / input_size))
        x2 = int(rect.right() * (orig_width / input_size))
        y2 = int(rect.bottom() * (orig_height / input_size))

        return (x1, y1, x2, y2)

    def draw_bounding_boxes(self, frame: np.ndarray, bboxes: np.ndarray) -> None:

        for x1, y1, x2, y2 in bboxes:

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

    def perform_detection(self, frame: np.ndarray, image_size: int = 600) -> np.ndarray:

        # Resize Image
        frame = cv.resize(frame, (image_size, image_size))

        # Perform Face Detection
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rects = self.classifier(rgb, 0)
        bboxes = [self.convert_rect_to_bb(
            r.rect, frame.shape[0], frame.shape[1], image_size) for r in rects
        ]

        # Draw Detected Faces Into Frame
        self.draw_bounding_boxes(frame, bboxes)

        return frame
