from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import itertools

import cv2
import time

from ir import IRThread
from vis import GPUThread

from ir.utils import overlay_bboxes as overlay_ir_bboxes
from vis.utils import draw_boxes as overlay_vis_bboxes, overlay_temps

import numpy as np


def draw_rectangle(arr):
    center = np.array([0.5, 0.5])
    wh = np.array([0.5, 0.5])

    p1 = np.array([center[0] - wh[0] / 2, center[1] - wh[1] / 2])

    p2 = np.array([center[0] + wh[0] / 2, center[1] + wh[1] / 2])

    # scale it to image size
    s = np.array(arr.shape[:2][::-1])

    p1 = tuple(np.array(p1 * s, dtype=np.int))
    p2 = tuple(np.array(p2 * s, dtype=np.int))

    cv2.rectangle(arr, p1, p2, (255, 255, 255), 3)


def exit_handler():
    print("exit handler called")
    gpu_thread.stop()
    ir_thread.stop()

    gpu_thread.join()
    ir_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    MAIN_MIN_LATENCY = 1 / 20  # run main thread at ~20Hz

    DISPLAY = True
    SAVE_FRAMES = False
    WIN_SIZE = (600, 450)
    FACE_BB_COLOR = (255, 255, 255)  # white
    EYES_BB_COLOR = (0, 255, 255)  # yellow
    LOG_DIR = "logs"

    APP_NAME = "AI Thermometer"
    IR_WIN_NAME = APP_NAME + ": IR frame"
    VIS_WIN_NAME = APP_NAME + ": VIS frame"

    gpu_thread = GPUThread(frame_size=WIN_SIZE)
    gpu_thread.start()

    ir_thread = IRThread(resize_to=WIN_SIZE)
    ir_thread.start()

    if SAVE_FRAMES:
        executor = ThreadPoolExecutor(max_workers=4)

    cv2.namedWindow(VIS_WIN_NAME)
    cv2.namedWindow(IR_WIN_NAME)
    cv2.moveWindow(IR_WIN_NAME, 0, WIN_SIZE[1])

    try:
        while gpu_thread.frame is None:
            print("Waiting for RGB frames")
            time.sleep(1)

        while ir_thread.frame is None:
            print("Waiting for IR frames")
            time.sleep(1)

        # main loop
        for i in itertools.count(start=0, step=1):

            time_start = time.monotonic()

            ir_arr = ir_thread.frame
            temps = ir_thread.temperatures

            rgb_arr = gpu_thread.frame
            dets = gpu_thread.detections

            rgb_arr_o = overlay_vis_bboxes(rgb_arr, dets)
            ir_arr_o = overlay_temps(rgb_arr, ir_arr, dets, temps)

            # draw rectangle to show the approx. IR overlay on VIS frame
            draw_rectangle(rgb_arr_o)

            # Show
            cv2.imshow(VIS_WIN_NAME, rgb_arr_o)
            cv2.imshow(IR_WIN_NAME, ir_arr_o)
            key = cv2.waitKey(1) & 0xFF

            # Save frames
            if SAVE_FRAMES:
                executor.submit(cv2.imwrite, f"{LOG_DIR}/frames/vis/{i:05d}.jpg", rgb_arr_o)
                executor.submit(cv2.imwrite, f"{LOG_DIR}/frames/ir/{i:05d}.jpg", ir_arr)

            # if the `q` key was pressed in the cv2 window, break from the loop
            if key == ord("q"):
                break

            main_latency = time.monotonic() - time_start
            print(
                f"GPU thread latency={gpu_thread._delay:.2f}    IR thread latency={ir_thread.latency:.2f}      Main thread latency={1000 * main_latency:.2f}"
            )

            time.sleep(max(0, MAIN_MIN_LATENCY - main_latency))

    finally:
        exit_handler()
