import cv2 as cv
import numpy as np
from FilePaths import Models
from BoxDef import BoxDef


class TinyYolo:
    def __init__(self, recognition=None):
        self.recognition = recognition
        self.classifier = cv.dnn.readNetFromDarknet(
            Models.PATH_TINYYOLO_CFG,
            Models.PATH_TINYYOLO_WEIGHTS
        )

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

    def perform_detection(self, frame: np.ndarray, image_size: int = 416) -> np.ndarray:

        # Resize Image
        frame = cv.resize(frame, (image_size, image_size))
        (H, W) = frame.shape[:2]

        # Output Layers
        ln = self.classifier.getLayerNames()
        ln = [ln[i - 1] for i in self.classifier.getUnconnectedOutLayers()]

        blob = cv.dnn.blobFromImage(
            frame, 1 / 255.0, (image_size, image_size), swapRB=True, crop=False)
        self.classifier.setInput(blob)

        # Perform Face Detection
        classifier_output = self.classifier.forward(ln)

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
        self.draw_bounding_boxes(frame, [boxes[i] for i in idxs])

        return frame
