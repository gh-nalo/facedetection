import os


def join_base(path: str) -> str:
    return os.path.join(Base.MODELS, path)


def join_results(path: str) -> str:
    return os.path.join(Base.RESULTS, path)


class Base:
    MODELS = "models/"
    RESULTS = "TimingResults/"


class Recognition:
    PATH_EMBEDDER = join_base(
        "face_recognition/openface.nn4.small2.v1.t7"
    )
    PATH_RECOGNIZER = join_base(
        "face_recognition/recognizer.pickle"
    )
    PATH_LABELENCODER = join_base(
        "face_recognition/labelencoder.pickle"
    )


class Models:
    PATH_VIOLAJONES = join_base(
        "ViolaJones/haarcascade_frontalface_alt.xml"
    )
    PATH_SSDRESNET_PROTO = join_base(
        "SSDResNet/deploy.prototxt.txt"
    )
    PATH_SSDRESNET_WEIGHTS = join_base(
        "SSDResNet/res10_300x300_ssd_iter_140000.caffemodel"
    )
    PATH_MMOD = join_base(
        "MMOD/mmod_human_face_detector.dat"
    )
    PATH_RFCN = join_base(
        "RFCN_Resnet101/rfcn_graph.pb"
    )
    PATH_RFCN_LITE = join_base(
        "RFCN_Resnet101/rfcn_640.tflite"
    )
    PATH_FASTERRCNN = join_base(
        "FasterRCNN_Inception_Resnet_v2/fasterrcnn_graph.pb"
    )
    PATH_FASTERRCNN_LITE = join_base(
        "FasterRCNN_Inception_Resnet_v2/faster_rcnn_640.tflite"
    )
    PATH_SSDMOBILENET = join_base(
        "SSDMobileNet/ssdmobilenet_graph.pb"
    )
    PATH_SSDMOBILENET_LITE = join_base(
        "SSDMobileNet/ssd_mobilenet_640.tflite"
    )
    PATH_TINYYOLO_CFG = join_base(
        "TinyYolo/tinyyolo.cfg"
    )
    PATH_TINYYOLO_WEIGHTS = join_base(
        "TinyYolo/tinyyolo.weights"
    )
    PATH_YOLOV2_CFG = join_base(
        "YoloV2/yolov2.cfg"
    )
    PATH_YOLOV2_WEIGHTS = join_base(
        "YoloV2/yolov2.weights"
    )
    PATH_YOLOV3_CFG = join_base(
        "YoloV3/yolov3.cfg"
    )
    PATH_YOLOV3_WEIGHTS = join_base(
        "YoloV3/yolov3.weights"
    )
