
from threading import Thread
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import itertools

import cv2
import time

from ir import IRThread
from vis import GPUThread

from ir.utils import overlay_bboxes as overlay_ir_bboxes 
from vis.utils import draw_boxes as overlay_vis_bboxes 

from flask import Flask, Response, render_template

def rgb_gen():
    while True:
        (success, encoded_image) = cv2.imencode(".jpg", vis_frame_w_overlay)

        if not success:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + \
            bytearray(encoded_image) + b'\r\n')

def ir_gen():
    while True:
        (success, encoded_image) = cv2.imencode(".jpg", ir_frame_w_overlay)

        if not success:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + \
            bytearray(encoded_image) + b'\r\n')

def make_flask_app():

    app = Flask(__name__)

    @app.route("/")
    def main_view():
        return render_template("main.html")

    @app.route("/ir")
    def ir_feed():
        return Response(ir_gen(),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    @app.route("/rgb")
    def rgb_feed():
        return Response(rgb_gen(),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

    return app

def exit_handler():
    print("exit handler called")
    gpu_thread.stop()
    ir_thread.stop()
    
    gpu_thread.join()
    ir_thread.join()

    # flask_thread.stop()
    app.do_teardown_appcontext()
    flask_thread.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    MAIN_MIN_LATENCY = 1/20 # cap the main thread at 20Hz

    DISPLAY = True
    WIN_SIZE = (800, 600)
    FACE_BB_COLOR = (255, 255, 255) # white
    EYES_BB_COLOR = (0,   255, 255) # yellow
    LOG_DIR = "logs"

    APP_NAME = "AI Thermometer"
    IR_WIN_NAME = APP_NAME + ": IR frame"
    VIS_WIN_NAME = APP_NAME + ": VIS frame" 

    # GPU Thread
    gpu_thread = GPUThread(frame_size=WIN_SIZE)
    gpu_thread.start()

    # IR processing thread
    ir_thread = IRThread(resize_to=WIN_SIZE, thr_temp=30)
    ir_thread.start()

    # Used for saving images
    executor = ThreadPoolExecutor(max_workers=4)

    # Webserver
    app = make_flask_app()
    flask_thread = Thread(target=app.run, kwargs = {"host": "0.0.0.0","port": 9000,"debug": True, "threaded": True, "use_reloader": False})
    flask_thread.start()

    vis_frame_w_overlay = None
    ir_frame_w_overlay = None

    # X-server interface
    cv2.namedWindow(VIS_WIN_NAME)
    cv2.namedWindow(IR_WIN_NAME)
    cv2.moveWindow(IR_WIN_NAME, WIN_SIZE[0], 0)
    
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

            ir_frame_w_overlay = overlay_ir_bboxes(ir_thread.frame, ir_thread.bboxes)
            vis_frame_w_overlay = overlay_vis_bboxes(gpu_thread.frame, gpu_thread.detections)

            # Show
            cv2.imshow(VIS_WIN_NAME, vis_frame_w_overlay)
            cv2.imshow(IR_WIN_NAME, ir_frame_w_overlay)
            key = cv2.waitKey(1) & 0xFF

            # Save frames
            executor.submit(cv2.imwrite, f"{LOG_DIR}/frames/vis/{i:05d}.jpg", vis_frame_w_overlay)
            executor.submit(cv2.imwrite, f"{LOG_DIR}/frames/ir/{i:05d}.jpg", ir_frame_w_overlay)
            
            # if the `q` key was pressed in the cv2 window, break from the loop
            if key == ord("q"):
                break

            main_latency =  time.monotonic() - time_start
            print(f"GPU thread latency={gpu_thread._delay:.2f}    IR thread latency={ir_thread.latency:.2f}      Main thread latency={1000 * main_latency:.2f}")
            
            time.sleep(max(0, MAIN_MIN_LATENCY - main_latency))

    finally:
        exit_handler()




