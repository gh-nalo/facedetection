import tensorflow as tf
import cv2 as cv
import numpy as np
from FilePaths import Models
from BoxDef import BoxDef


class SSDMobileNetLite:
    def __init__(self, recognition=None):
        self.recognition = recognition
        self.min_confidence = 0.5

        self.interpreter = tf.lite.Interpreter(Models.PATH_SSDMOBILENET_LITE)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def draw_bounding_boxes(self, frame: np.ndarray, bboxes: np.ndarray) -> None:

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
        frame = cv.resize(frame, (self.width, self.height))

        input_data = np.expand_dims(frame, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - 175.5) / 175.5

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Perform Face Detection
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])[0]
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        im_height, im_width, _ = frame.shape

        bboxes = []

        for i in range(boxes.shape[0]):
            if scores[i] >= self.min_confidence:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x1, y1, x2, y2 = int(left), int(top), int(
                    right - left), int(bottom - top)

                bboxes.append([x1, y1, x2, y2])

        # Draw Detected Faces Into Frame
        self.draw_bounding_boxes(frame, bboxes)

        return frame
