import pickle
import cv2 as cv
import numpy as np
import os

BASE_PATH = "./faceRecognition/"

PATH_EMBEDDER = os.path.join(BASE_PATH, "openface.nn4.small2.v1.t7")
PATH_RECOGNIZER = os.path.join(BASE_PATH, "recognizer.pickle")
PATH_LABELENCODER = os.path.join(BASE_PATH, "labelencoder.pickle")


class FaceRecognition:

    def __init__(self):
        self.embedder = cv.dnn.readNetFromTorch(PATH_EMBEDDER)

        with open(PATH_RECOGNIZER, 'rb') as handle:
            self.recognizer = pickle.load(handle)

        with open(PATH_LABELENCODER, 'rb') as handle:
            self.label_encoder = pickle.load(handle)

    def recognize_face(self, frame, x1, y1, x2, y2):
        '''
            Returns The Label For an Image Bbox Given After Performing Face Recognition
        '''
        # Limit Image To Face
        face = frame[y1:y2, x1:x2]

        # Set Input
        face_blob = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(face_blob)

        # Get Embedding
        embedding = self.embedder.forward()

        # Classification
        predictions = self.recognizer.predict_proba(embedding)[0]
        max_index = np.argmax(predictions)
        probability = predictions[max_index]
        name = self.label_encoder.classes_[max_index]

        return f"{name}: {probability * 100:.2f}%"
