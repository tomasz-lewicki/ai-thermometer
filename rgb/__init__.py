from threading import Thread
import time, os
import numpy as np
import cv2
from .ssd import SsdDetector
from .camera import make_imx219_capture


class RGBThread(Thread):
    def __init__(self, stream=None, frame_size=(1088, 816)):

        super(RGBThread, self).__init__()

        parent_dir_pth = os.path.dirname(os.path.abspath(__file__))

        self._detector = SsdDetector(
            prototxt_file_pth=parent_dir_pth + "/ssd/caffe/deploy.prototxt.txt",
            caffe_model_pth=parent_dir_pth
            + "/ssd/caffe/res10_300x300_ssd_iter_140000.caffemodel",
        )

        self._stream = make_imx219_capture(0)
        self._delay = -1
        self._detections = None
        self._running = True
        self._frame = None

    def run(self):

        while self._running:
            loop_start = time.monotonic()
            ret, self._frame = self._stream.read()
            # ret, _frame = self._stream.retrieve()

            if ret is False:
                print("WARN: no frame")
                time.sleep(1)
                continue

            self._detections = self._detector(self._frame)
            self._delay = 1000 * (time.monotonic() - loop_start)

        print("releasing")
        self._stream.release()

    def get_faces(self):
        """
        :ret list of x,y positions of faces in image _frame:
        """

        return self._detections

    def stop(self):
        self._running = False

    @property
    def frame(self):
        return self._frame

    @property
    def detections(self):
        return self._detections
