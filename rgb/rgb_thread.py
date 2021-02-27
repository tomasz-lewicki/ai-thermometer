import time
from threading import Thread

import cv2
import numpy as np

from .camera import make_imx219_capture
from .retinaface import RetinaFaceDetector
from .ssd import SsdDetector


class RGBThread(Thread):
    def __init__(self, model="retinaface"):

        super(RGBThread, self).__init__()

        if model == "SSD":
            self._detector = SsdDetector()
        elif model == "retinaface":
            self._detector = RetinaFaceDetector()
        else:
            raise ValueError(f"{self._detector} is not a valid argument.")

        self._stream = make_imx219_capture(0)
        self._delay = -1
        self._scores, self._boxes, self._landms = [], [], []
        self._running = True
        self._frame = np.zeros((1024, 768), np.uint8)
        self._frame_small = np.zeros((400, 300), np.uint8)

    def run(self):

        while self._running:

            loop_start = time.monotonic()
            ret, self._frame = self._stream.read()

            frame_small = cv2.resize(self._frame, (400, 300))

            if ret is False:
                print("WARN: no frame")
                time.sleep(1)
                continue

            self._scores, self._boxes, self._landms = self._detector(frame_small)
            self._delay = 1000 * (time.monotonic() - loop_start)

        print("releasing")
        self._stream.release()

    def get_detections(self):
        return self._scores, self._boxes, self._landms

    def stop(self):
        self._running = False

    @property
    def frame(self):
        return self._frame

    @property
    def detections(self):
        return self._detections
