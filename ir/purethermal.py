from threading import Thread
from queue import Queue
import numpy as np
import cv2, time

# IR camera
from .libuvc_wrapper import *
from .utils import ktoc, resize, normalize, crop_telemetry, detect_ir, drop_small_bboxes


def uvc_init(ctx):
    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(res)

def find_device(ctx, dev):
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
        print("uvc_find_device error")
        exit(res)

def open_device(dev, devh):
    res = libuvc.uvc_open(dev, byref(devh))
    if res < 0:
        print(f"uvc_open error {res}")
        exit(res)
    else:
        print("device opened!")

def check_frame_formats(frame_formats):
    if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

def start_streaming(devh, ctrl, ptr_callback):
    res = libuvc.uvc_start_streaming(
        devh, byref(ctrl), ptr_callback, None, 0
    )
    if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

def start_pt2(dev, devh, ctx, q):
    # initialize Pure Thermal 2 board
    ctrl = uvc_stream_ctrl()

    uvc_init(ctx)
    find_device(ctx, dev)
    open_device(dev, devh)

    print_device_info(devh)
    print_device_formats(devh)

    frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
    check_frame_formats(frame_formats)

    libuvc.uvc_get_stream_ctrl_format_size(
        devh,
        byref(ctrl),
        UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth,
        frame_formats[0].wHeight,
        int(1e7 / frame_formats[0].dwDefaultFrameInterval),
    )
    
    PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)
    start_streaming(devh, ctrl, PTR_PY_FRAME_CALLBACK)

    return PTR_PY_FRAME_CALLBACK

def py_frame_callback(frame, userptr):
    array_pointer = cast(
        frame.contents.data,
        POINTER(c_uint16 * (frame.contents.width * frame.contents.height)),
    )
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
        frame.contents.height, frame.contents.width
    )

    assert frame.contents.data_bytes == (2 * frame.contents.width * frame.contents.height)

    if not q.full():
        q.put(data)

def setup():
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(res)

    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
        print("uvc_find_device error")
        exit(res)

    res = libuvc.uvc_open(dev, byref(devh))
    if res < 0:
        print(f"uvc_open error {res}")
        exit(res)

    print("device opened!")

    print_device_info(devh)
    print_device_formats(devh)

    frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
    if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

    libuvc.uvc_get_stream_ctrl_format_size(
        devh,
        byref(ctrl),
        UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth,
        frame_formats[0].wHeight,
        int(1e7 / frame_formats[0].dwDefaultFrameInterval),
    )

    PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

    res = libuvc.uvc_start_streaming(
        devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0
    )
    if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)


    return ctx, dev, devh, ctrl

class IRThread(Thread):
    def __init__(self, bufsize=2, resize_to=(800,600), thr_temp=28):
        super(IRThread, self).__init__()
        
        self._ctx = POINTER(uvc_context)()
        self._dev = POINTER(uvc_device)()
        self._devh = POINTER(uvc_device_handle)()
        self._cb_ptr = start_pt2(self._dev, self._devh, self._ctx, q)

        self._thr_temp = thr_temp
        self._resize_to = resize_to

        self._frame_raw = None
        self._frame_upscaled = None
        self._frame_normalized = None
        self._bboxes = None
        self._latency = -1.0
        self._running = True

    def run(self):
        try:
            while self._running:
                start_time = time.monotonic()
                # get frame
                frame = q.get(True, 500)
                
                # processing
                frame = crop_telemetry(frame)
                frame = ktoc(frame) # 16-bit Kelvin to deg C
                upscaled = resize(frame, size=self._resize_to)
                normalized = normalize(upscaled.copy())

                # detections
                bboxes_all = detect_ir(upscaled, self._thr_temp)
                bboxes_good = drop_small_bboxes(bboxes_all, min_size=1000)

                # save members
                self._frame_raw = frame
                self._bboxes = bboxes_good
                self._frame_upscaled = upscaled
                self._frame_normalized = normalized

                self._latency = 1000 * (time.monotonic() - start_time)

        finally:
            self._exit_handler()

    def _exit_handler(self):
        libuvc.uvc_stop_streaming(self._devh)
        libuvc.uvc_unref_device(self._dev)
        libuvc.uvc_exit(self._ctx)


    @property
    def frame(self):
        # TODO: add locks
        return self._frame_normalized 

    @property
    def temperatures(self, upscaled=True):
        if upscaled:
            return self._frame_upscaled
        else:
            return self._frame_raw
    
    @property
    def bboxes(self):
        return self._bboxes
        
    @property
    def latency(self):
        return self._latency

    def stop(self):
        self._running = False

q = Queue(2) # not ideal, but q must be global for the callback function to have access to it

if __name__ == "__main__":
    
    # example usage

    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()

    p = start_pt2(dev, devh, ctx, q)

    try:
        while True:
            data = q.get(True, 500)
            print(ktoc(data))
    finally:
        libuvc.uvc_stop_streaming(devh)
        libuvc.uvc_unref_device(dev)
        libuvc.uvc_exit(ctx)

