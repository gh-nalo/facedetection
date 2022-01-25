import cv2 as cv
import numpy as np
import tensorflow as tf
from FilePaths import Models
from BoxDef import BoxDef


class FasterRCNN:
    def __init__(self, recognition=None):
        self.recognition = recognition
        self.detection_graph = tf.Graph()
        self.min_confidence = 0.8

        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()

            with tf.compat.v2.io.gfile.GFile(Models.PATH_FASTERRCNN, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.compat.v1.Session(graph=self.detection_graph)

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
                    BoxDef.BoxDef.TEXT_COLOR,
                    BoxDef.TEXT_WIDTH
                )

            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BoxDef.BOX_COLOR,
                BoxDef.BOX_WIDTH,
            )

    def perform_detection(self, frame: np.ndarray, image_size: int = 640) -> np.ndarray:

        # Resize Image
        frame = cv.resize(frame, (image_size, image_size))

        # Expand Image Dimensions For Model ([1, None, None, 3])
        image_np_expanded = np.expand_dims(frame, axis=0)

        # Image
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Perform Face Detection
        (boxes, scores, _, _) = self.session.run(
            [
                self.detection_graph.get_tensor_by_name(
                    'detection_boxes:0'),  # Object Bounding Box
                self.detection_graph.get_tensor_by_name(
                    'detection_scores:0'),  # Confidence
                self.detection_graph.get_tensor_by_name(
                    'detection_classes:0'),  # Id Of Detected Class
                self.detection_graph.get_tensor_by_name(
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
            if scores[i] >= self.min_confidence:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x, y, w, h = int(left), int(top), int(
                    right - left), int(bottom - top)

                bboxes.append([x, y, x + w, y + h])

        # Draw Detected Faces Into Frame
        self.draw_bounding_boxes(frame, bboxes)

        return frame
