from util.WebcamVideoStream import WebcamVideoStream
from util.FaceRecognition import FaceRecognition
from util.SaveTimings import SaveTimings
import time
import cv2 as cv
import argparse


def main(detector, name, n_frames=None, image_size=None) -> None:

    # Video Capture
    video_capture = WebcamVideoStream(0).start()

    # Saving Times
    timer = SaveTimings(name, save_results=True)

    # Count Frames Per Second
    starting_time = time.time()
    fps_counter = 0
    n_frames = n_frames if n_frames else 200

    # Detection Loop
    while True:

        # Capture Frame
        frame_captured, frame = video_capture.read()

        # Stop if There is no Image
        if not frame_captured:
            print("No image returned from camera.")
            break

        # Face Detection & Recognition
        if image_size:
            frame = detector.perform_detection(frame, image_size=image_size)

        else:
            frame = detector.perform_detection(frame)

        # Show Results
        cv.imshow(name, frame)

        # Count Average Frames Per Second
        fps_counter += 1

        if (time.time() - starting_time) > 1.0:

            fps = fps_counter / (time.time() - starting_time)

            if timer.new_value(fps) > n_frames:
                timer.save_results()
                break

            print(f"FPS: {fps:.4f}")

            fps_counter = 0
            starting_time = time.time()

        # Cancel Program With "Q"
        if cv.waitKey(1) & 0xFF == ord('q'):
            timer.save_results()
            break

    # Cleanup
    cv.destroyAllWindows()
    video_capture.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm To Run Inference On"
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=False,
        help="Image Size To Run On"
    )

    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        required=False,
        help="N_Frames To Run For"
    )

    args = vars(parser.parse_args())
    name = args["algorithm"]
    size = args["size"]
    n_frames = args["frames"]

    recognition = FaceRecognition()

    # TODO :: In Python 3.10 replace with match/case
    if name == "ViolaJones":
        from detectors.ViolaJones import ViolaJones
        detector = ViolaJones(recognition)

    elif name == "SSDResNet":
        from detectors.SSDResNet import SSDResNet
        detector = SSDResNet(recognition)

    elif name == "DLibHogSVM":
        from detectors.DLibHogSVM import DLibHogSVM
        detector = DLibHogSVM(recognition)

    elif name == "DLibMMOD":
        from detectors.DLibMMOD import DLibMMOD
        detector = DLibMMOD(recognition)

    elif name == "MTCNN":
        from detectors.MTCNN import MTCNN
        detector = MTCNN(recognition)

    elif name == "SSDMobileNet":
        from detectors.SSDMobileNet import SSDMobileNet
        detector = SSDMobileNet(recognition)

    elif name == "TinyYolo":
        from detectors.TinyYolo import TinyYolo
        detector = TinyYolo(recognition)

    elif name == "YoloV2":
        from detectors.YoloV2 import YoloV2
        detector = YoloV2(recognition)

    elif name == "YoloV3":
        from detectors.YoloV3 import YoloV3
        detector = YoloV3(recognition)

    elif name == "RFCN":
        from detectors.RFCN import RFCN
        detector = RFCN(recognition)

    elif name == "FasterRCNN":
        from detectors.FasterRCNN import FasterRCNN
        detector = FasterRCNN(recognition)

    elif name == "RFCNLite":
        from detectors.RFCNLite import RFCNLite
        detector = RFCNLite(recognition)

    elif name == "SSDMobileNetLite":
        from detectors.SSDMobileNetLite import SSDMobileNetLite
        detector = SSDMobileNetLite(recognition)

    elif name == "FasterRCNNLite":
        from detectors.FasterRCNNLite import FasterRCNNLite
        detector = FasterRCNNLite(recognition)

    else:
        print("Error: Mismatch on Algorithm")
        exit()

    main(detector, name, n_frames, args["size"])
