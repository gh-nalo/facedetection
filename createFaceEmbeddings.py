from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import cv2 as cv
import os

# Filepaths
PATH_DATASET = "./faceRecognition/dataset"
PATH_EMBEDDINGS = "./faceRecognition/embeddings.pickle"
PATH_RECOGNIZER = "./faceRecognition/recognizer.pickle"
PATH_LABELENCODER = "./faceRecognition/labelencoder.pickle"

PATH_PROTOTXT = "./__models/02_SSDResNet/deploy.prototxt.txt"
PATH_MODEL = "./__models/02_SSDResNet/res10_300x300_ssd_iter_140000.caffemodel"

PATH_EMBEDDER = "./faceRecognition/openface.nn4.small2.v1.t7"

# Minimum Confidence For Face Detection In Dataset
MIN_CONFIDENCE = 0.5


def get_image_paths():
    items = os.listdir(PATH_DATASET)

    image_paths = []

    for item in items:
        current_item = os.path.join(PATH_DATASET, item)

        if os.path.isdir(current_item):
            files = os.listdir(current_item)

            image_paths.extend([os.path.join(current_item, file)
                               for file in files])

    return image_paths


def main():
    # Face Detector
    print("...Loading Face Detector")
    detector = cv.dnn.readNetFromCaffe(PATH_PROTOTXT, PATH_MODEL)

    # Face Recognizer
    print("...Loading Face Recognizer")
    embedder = cv.dnn.readNetFromTorch(PATH_EMBEDDER)  # Source :: OpenFace

    # Images
    image_paths = get_image_paths()

    known_embeddings = []
    known_names = []

    # Loop Through Images
    for (i, image_path) in enumerate(image_paths):

        # Extract person name from folder name
        print(f"...Processing image {i + 1}/{len(image_paths)}")

        name = image_path.split(os.path.sep)[-2]

        # Resize Image
        image = cv.imread(image_path)
        image = cv.resize(image, (640, 640))
        (h, w) = image.shape[:2]

        # Create Input
        image_blob = cv.dnn.blobFromImage(
            cv.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(image_blob)

        # Detect Faces
        detections = detector.forward()

        # Get Embeddings If Face Is Detected
        if len(detections) > 0:

            # One face per image
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Take highest confidence face
            if confidence > MIN_CONFIDENCE:

                # Bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Limit image to face
                face = image[y1:y2, x1:x2]
                (face_height, face_width) = face.shape[:2]

                # Ensure sufficient face width and height
                if face_width < 20 or face_height < 20:
                    continue

                face_blob = cv.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)

                # Get Embeddings
                vec = embedder.forward()

                # Store Embeddings And Corresponding Names
                known_names.append(name)
                known_embeddings.append(vec.flatten())

    data = {"embeddings": known_embeddings, "names": known_names}

    # Write Embeddings Into File
    print("...Saving Embeddings")

    with open(PATH_EMBEDDINGS, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Encode Labels
    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(data["names"])

    # Train SVC Model For Face Recognition (Parameters Could be Optimised With GridSearch)
    print("...Training SVC Model")

    # recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer = SVC(
        C=1,
        kernel='rbf',
        probability=True,
        gamma=2
    )  # OpenFace Recommendation
    recognizer.fit(data["embeddings"], labels)

    # Write Face Recognition Model Into File
    print("...Saving Recognizer")

    with open(PATH_RECOGNIZER, 'wb') as handle:
        pickle.dump(recognizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Write LabelEncoder Into File
    print("...Saving LabelEncoder")

    with open(PATH_LABELENCODER, 'wb') as handle:
        pickle.dump(labelEncoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nFinished.")


if __name__ == "__main__":
    main()
