import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from ir import IRThread
from ui import make_ir_view, make_rgb_view
from vis import GPUThread


def get_config(file_path=None):
    HZ_CAP = 20
    LOG_DIR = "logs"
    IR_WIN_NAME = "IR view"
    VIS_WIN_NAME = "RGB view"

    VIS_BBOX_COLOR = (0, 0, 255)  # red
    IR_BBOX_COLOR = (0, 255, 0)  # green

    IR_WIN_SIZE = (400, 300)
    VIS_WIN_SIZE = (400, 300)

    SAVE_FRAMES = True
    SHOW_DISPLAY = True
    MAX_FILE_QUEUE = 10

    FRAME_SIZE = (1024, 768)

    X_DISPLAY_ADDR = ":0"

    return (
        HZ_CAP,
        LOG_DIR,
        SHOW_DISPLAY,
        SAVE_FRAMES,
        MAX_FILE_QUEUE,
        FRAME_SIZE,
        IR_WIN_NAME,
        IR_WIN_SIZE,
        VIS_WIN_NAME,
        VIS_WIN_SIZE,
        X_DISPLAY_ADDR,
        VIS_BBOX_COLOR,
        IR_BBOX_COLOR,
    )


def exit_handler():
    print("exit handler called")
    gpu_thread.stop()
    ir_thread.stop()

    gpu_thread.join()
    ir_thread.join()

    cv2.destroyAllWindows()


def setup_display(display_addr):
    if os.environ.get("DISPLAY") is None:
        os.environ["DISPLAY"] = display_addr
    elif X_DISPLAY_ADDR:
        print("WARN: Using $DISPLAY from environment, not from config")

    cv2.namedWindow(VIS_WIN_NAME)
    cv2.namedWindow(IR_WIN_NAME)
    cv2.moveWindow(IR_WIN_NAME, VIS_WIN_SIZE[1], 0)


def mainloop():
    # main loop
    for i in itertools.count(start=0, step=1):

        time_start = time.monotonic()

        ir_raw = ir_thread.raw
        ir_arr = ir_thread.frame
        temps = ir_thread.temperatures

        rgb_arr = gpu_thread.frame
        dets = gpu_thread.detections

        rgb_view = make_rgb_view(
            rgb_arr, dets, VIS_WIN_SIZE, bb_color=VIS_BBOX_COLOR
        )

        ir_view = make_ir_view(
            rgb_arr, ir_arr, dets, temps, IR_WIN_SIZE, bb_color=IR_BBOX_COLOR
        )

        # Show
        if SHOW_DISPLAY:
            cv2.imshow(VIS_WIN_NAME, rgb_view)
            cv2.imshow(IR_WIN_NAME, ir_view)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed in the cv2 window, break from the loop
            if key == ord("q"):
                break

        # Save frames
        if SAVE_FRAMES:
            if executor._work_queue.qsize() > MAX_FILE_QUEUE:
                print(
                    "Error: Too many files in file queue. Not saving frames from this iteration."
                )
            else:
                executor.submit(
                    cv2.imwrite, f"{LOG_DIR}/frames/vis/{i:05d}.jpg", rgb_view
                )
                executor.submit(
                    cv2.imwrite, f"{LOG_DIR}/frames/ir/{i:05d}.png", ir_view
                )

        main_latency = time.monotonic() - time_start
        print(
            f"GPU thread latency={gpu_thread._delay:.2f}    IR thread latency={ir_thread.latency:.2f}      Main thread latency={1000 * main_latency:.2f}"
        )

        time.sleep(max(0, 1 / HZ_CAP - main_latency))


if __name__ == "__main__":

    (
        HZ_CAP,
        LOG_DIR,
        SHOW_DISPLAY,
        SAVE_FRAMES,
        MAX_FILE_QUEUE,
        FRAME_SIZE,
        IR_WIN_NAME,
        IR_WIN_SIZE,
        VIS_WIN_NAME,
        VIS_WIN_SIZE,
        X_DISPLAY_ADDR,
        VIS_BBOX_COLOR,
        IR_BBOX_COLOR,
    ) = get_config()

    gpu_thread = GPUThread(frame_size=FRAME_SIZE)
    gpu_thread.start()

    ir_thread = IRThread(resize_to=FRAME_SIZE)
    ir_thread.start()

    if SAVE_FRAMES:
        executor = ThreadPoolExecutor(max_workers=4)

    if SHOW_DISPLAY:
        setup_display(X_DISPLAY_ADDR)

    try:
        while gpu_thread.frame is None:
            print("Waiting for RGB frames")
            time.sleep(1)

        while ir_thread.frame is None:
            print("Waiting for IR frames")
            time.sleep(1)

        mainloop()

    finally:
        exit_handler()
