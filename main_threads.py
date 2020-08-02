
from threading import Thread
from queue import Queue

import cv2
import numpy as np

import time

import ir

class GPUThread(Thread):

    def __init__(self, stream):

        super(GPUThread, self).__init__()
        self._net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self._stream = stream
        self._delay = -1
        self._detections = None
        self._running = True
        self._frame = None

    def run(self):

        while self._running:
            loop_start = time.monotonic()
            ret, self._frame = stream.read()
            # ret, _frame = self._stream.retrieve()
            if ret is False:
                print("WARN: no frame")
                time.sleep(1)
                continue

            # detect
            blob = cv2.dnn.blobFromImage(
                cv2.resize(self._frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
                )
            self._net.setInput(blob)
            detections = self._net.forward()

            self._detections = np.squeeze(detections)
            self._delay = 1000*(time.monotonic()-loop_start)
        
        self._stream.release()


    def get_faces(self):
        """
        :ret list of x,y positions of faces in image _frame:
        """

        return self._detections

    def stop(self):
        self._running = False

    @property
    def frame():
        return self._frame

# class CameraThread(Thread):

#     def __init__(self, stream):
#         """
#         :param stream: cv2 stream object
#         """
#         super(CameraThread, self).__init__()

#         self._running = True
#         self.__frame_queue = Queue(maxsize=2)
#         self._stream = stream
        


#     def read(self):
#         return self.__frame_queue.pop()


#     def run(self):

#         while self._running:
#             ret, arr = self._stream.read()
#             self.__frame_queue.push(arr)




def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=1024,
    display_height=768,
    framerate=10,
    flip_method=2,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def make_stream():

    stream = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=3264,
            capture_height=2464,
            display_width=1088,
            display_height=816,
            framerate=21,
            flip_method=2,
        ),
        cv2.CAP_GSTREAMER,
    )

    return stream


if __name__ == "__main__":

    stream = make_stream()
    
    gpu_thread = GPUThread(stream)

    gpu_thread.start()

    while True:
        time.sleep(.05)

        if gpu_thread._frame is None:
            print("Waiting for frames")
            time.sleep(1)
            continue
        else:
            print(f"GPU thread delay={gpu_thread._delay:.2f}")
        
        cv2.imshow("_frame", gpu_thread._frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            gpu_thread.stop()
            gpu_thread.join()
            break


    # ct = CameraThread(stream)

    # ct.start()


