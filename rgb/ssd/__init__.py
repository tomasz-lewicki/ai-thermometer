import numpy as np
import cv2


class SsdDetector:
    def __init__(self, prototxt_file_pth, caffe_model_pth):

        self._in_size = (300, 300)

        print("Loading weights from file...")
        self._net = cv2.dnn.readNetFromCaffe(prototxt_file_pth, caffe_model_pth)
        print("Weights loaded!")

        if hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("cv2.dnn.DNN_BACKEND_CUDA not available!")

        print("Running first net inference...")
        test_arr = arr = np.ones((768, 1024, 3), dtype=np.uint8)
        self(test_arr)
        print("Detector initialized!")

    def __call__(self, arr):

        blob = cv2.dnn.blobFromImage(
            cv2.resize(arr, self._in_size), 1.0, self._in_size, (104.0, 177.0, 123.0)
        )

        self._net.setInput(blob)
        detections = self._net.forward()

        return np.squeeze(detections)