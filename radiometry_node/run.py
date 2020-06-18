import time
import cv2
import numpy as np
from queue import Queue

# IR camera
from libuvc_wrapper import *

# ROS2
import rclpy
from std_msgs.msg import String
from rclpy.node import Node

import jsonpickle # to parse numpy array

BUF_SIZE = 2
q = Queue(BUF_SIZE)


def py_frame_callback(frame, userptr):

    array_pointer = cast(
        frame.contents.data,
        POINTER(c_uint16 * (frame.contents.width * frame.contents.height)),
    )
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
        frame.contents.height, frame.contents.width
    )  # no copy

    # data = np.fromiter(
    #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
    # ).reshape(
    #   frame.contents.height, frame.contents.width, 2
    # ) # copy

    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    if not q.full():
        q.put(data)


PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktoc(val):
    return (val - 27315) / 100.0


if __name__ == "__main__":

    rclpy.init(args=None)
    node = rclpy.create_node('radiometry')
    publisher = node.create_publisher(String, 'heatmap', 10)


    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(res)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(res)

        try:
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

            res = libuvc.uvc_start_streaming(
                devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0
            )
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            try:
                start = time.time()
                while True:
                    data = q.get(True, 500)
                    if data is None:
                        break

                    temp_map = ktoc(data)

                    m = String()
                    m.data = jsonpickle.encode(temp_map)
                    publisher.publish(m)


            finally:
                libuvc.uvc_stop_streaming(devh)

            print("done")
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)
        
        node.destroy_node()
        rclpy.shutdown()
