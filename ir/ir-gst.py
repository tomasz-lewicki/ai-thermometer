from threading import Thread
import numpy as np
import cv2, time


def lepton3_pipeline():
    return "v4l2src device=/dev/video1 ! video/x-raw,format=GRAY16_LE ! videoconvert ! appsink max-buffers=1 drop=true "

def make_ir_stream():
    pipeline = lepton3_pipeline()
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

def ktoc(val):
    # Kelvin to Celsius
    return (val - 27315) / 100.0

def resize(ir_arr, size):
    ir_arr = cv2.resize(ir_arr, size)
    return ir_arr

def normalize(ir_arr):
    ir_arr_n = cv2.normalize(ir_arr, None, 0, 255, cv2.NORM_MINMAX)  # normalize to 0-255
    ir_arr_n = ir_arr_n.astype(np.uint8)  # convert to uint8
    return ir_arr_n

def crop_telemetry(ir_arr):
    return ir_arr[:-2,:]





class IRThread(Thread):
    def __init__(self, stream=make_ir_stream(), size=(800, 600)):
        super(IRThread, self).__init__()
        self._stream = stream
        self._size = size
        self._frame = None
        self._running = True

    def run(self):
        while self._running:
            ret, frame = self._stream.read()

            # IR processing
            frame = crop_telemetry(frame)
            # frame = ktoc(frame)
            print(frame)
            upscaled = resize(frame, size=self._size)
            normalized = normalize(upscaled)

            self._frame_raw = frame
            self._frame_upscaled = upscaled
            self._frame_normalized = normalized

            # print(upscaled)

            time.sleep(0.05)  # Lepton 3 is 9Hz, no need to loop more often

    @property
    def frame(self):
        # TODO: add locks
        return self._frame_normalized 

    def stop(self):
        self._stream.release()
        self._running = False
