from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import itertools 
import numpy as np
import cv2
import time

# For ROS core
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# For ROS2 messages
from std_msgs.msg import String
import jsonpickle

from aithermometer.visible.capture import gstreamer_pipeline

class HeatmapClient(Node):
    def __init__(self):
        super().__init__("heatmap_listener")
        self.hm = None
        self.sub = self.create_subscription(String, "heatmap", self.hm_callback, 10)

    def hm_callback(self, msg):
        self.hm = jsonpickle.decode(msg.data)
        # print(f"Persons temp. : {round(self.hm.max(), 4)} deg C")


def ir_processing(ir_arr):
    ir_arr = cv2.resize(ir_arr, SIZE)  # upscale
    return ir_arr

def normalize_ir(ir_arr):
    ir_arr_normalized = cv2.normalize(ir_arr, None, 0, 255, cv2.NORM_MINMAX)  # normalize to 0-255
    ir_arr_normalized = ir_arr_normalized.astype(np.uint8)  # convert to uint8
    return ir_arr_normalized

def rgb_processing(rgb_arr):
    # RGB processing

    T = np.array([[1.1, 0.0, 0], [0.0, 1.1, 0]])

    rgb_arr = cv2.resize(rgb_arr, SIZE)
    rgb_arr = cv2.warpAffine(
        rgb_arr, T, (rgb_arr.shape[1], rgb_arr.shape[0])
    )  # transform

    return rgb_arr



def images_bad(rgb_arr, ir_arr):
    bad = False

    if ir_arr is None:
        print("Waiting for IR frames...")
        bad = True
    if rgb_arr is None:
        print("Waiting for RGB frames...")
        bad = True

    return bad

def camera_loop():

    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=3264,
            capture_height=2464,
            display_width=SIZE[0],
            display_height=SIZE[1],
            framerate=21,
            flip_method=2,
        ),
        cv2.CAP_GSTREAMER,
    )

    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
    )

    frames = []

    for i in itertools.count(start=0, step=1):

        t = time.monotonic()

        ir_arr = irclient.hm # get IR frame
        ret_val, rgb_arr = cap.read() # get RGB frame
        
        # check if we got frames from cameras
        if images_bad(ir_arr, rgb_arr):
            time.sleep(0.1)
            continue

        # preprocessing
        rgb_arr = rgb_processing(rgb_arr)
        gray_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2GRAY)
        # gray_arr_3ch = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)

        ir_arr = ir_processing(ir_arr)

        # face detection
        detections = detect_faces_haar(gray_arr, face_cascade, eye_cascade)

        # get temperatures
        temps = get_temps(ir_arr, detections)
        print(temps)

        # bbox overlay
        rgb_arr = overlay_bboxes(rgb_arr, detections, temps)
        # composed_arr = compose_vis_ir(rgb_arr, ir_arr_n)

        ir_cmap = apply_cmap(ir_arr)
        if DISPLAY:
            cv2.imshow(APP_NAME+": RGB view", rgb_arr)
            cv2.imshow(APP_NAME+": IR view", ir_cmap)
            cv2.waitKey(1)

        saving_executor.submit(cv2.imwrite, f"frames_rgb/{i:05d}.jpg", rgb_arr)
        saving_executor.submit(cv2.imwrite, f"frames_ir/{i:05d}.jpg", ir_cmap)
 
        delay = round(time.monotonic()-t, 2)
        print(f"Loop FPS {1/delay}")

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # config
    # SIZE = (1024, 768)
    SIZE = (800, 600)
    DISPLAY = True
    APP_NAME = "AI Thermometer"

    FACE_BB_COLOR = (255, 255, 255) # white
    EYES_BB_COLOR = (0,   255, 255) # yellow

    rclpy.init(args=None)

    # ROS publisher
    # node = rclpy.create_node("minimal_publisher")
    # publisher = node.create_publisher(String, "rgb_detections", 10)

    # ROS listener
    irclient = HeatmapClient()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(irclient)

    t = Thread()
    t.run = executor.spin
    t.start()

    # Frame saver 
    saving_executor = ThreadPoolExecutor(max_workers=1)

    try:
        camera_loop()

    finally:
        executor.shutdown()
        irclient.destroy_node()
        t.join()
        rclpy.shutdown()
