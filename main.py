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


def gstreamer_pipeline(
    capture_width=3264,
    capture_height=2464,
    display_width=1024,
    display_height=768,
    framerate=21,
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
    ir_arr[ir_arr < 30] = 0  # clip at 30deg C
    ir_arr_normalized = cv2.normalize(
        ir_arr, None, 0, 255, cv2.NORM_MINMAX
    )  # normalize to 0-255
    ir_arr_normalized = ir_arr_normalized.astype(np.uint8)  # convert to uint8

    return ir_arr_normalized


def rgb_processing(rgb_arr):
    # RGB processing

    T = np.array([[1.1, 0.0, 0], [0.0, 1.1, 0]])

    rgb_arr = cv2.resize(rgb_arr, SIZE)
    rgb_arr = cv2.warpAffine(
        rgb_arr, T, (rgb_arr.shape[1], rgb_arr.shape[0])
    )  # transform
    rgb_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2GRAY)

    return rgb_arr

def detect_faces(gray_arr, face_clf, eye_clf):
    """
    :param gray_arr: np grayscale aray with the image
    :param fc: cv2.CascadeClassifier object for face classification
    :param ec: cv2.CascadeClassifier object for eye classification
    """

    faces = face_clf.detectMultiScale(gray_arr, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(gray_arr, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi = gray_arr[y : y + h, x : x + w]
        eyes = eye_clf.detectMultiScale(roi)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 2)

    return gray_arr


def compose_channels(gray, ir):
    
    blue = gray
    red = ir
    green = np.zeros_like(blue)

    composed_arr = np.stack([blue, green, red], axis=-1)

    return composed_arr

def camera_loop():

    cap = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=3264,
            capture_height=2464,
            display_width=1024,
            display_height=768,
            framerate=21,
            flip_method=2,
        ),
        cv2.CAP_GSTREAMER,
    )
    window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
    )

    frames = []

    for i in itertools.count(start=0, step=1):

        t = time.monotonic()

        # get RGB frame
        # get IR frame
        ir_arr = irclient.hm 
        ret_val, rgb_arr = cap.read()
        gray_arr = rgb_processing(rgb_arr)
        

        # check if we got IR frames
        if ir_arr is None:
            print("Waiting for IR frames...")
            time.sleep(0.1)
            continue
        
        if rgb_arr is None:
            print("Waiting for RGB frames...")
            time.sleep(0.1)
            continue

        ir_arr = ir_processing(ir_arr)

        composed_arr = compose_channels(gray_arr, ir_arr)

        cv2.imshow(APP_NAME, composed_arr)

        cv2.waitKey(1)
        
        saving_executor.submit(cv2.imwrite, f"frames/{i:05d}.jpg", composed_arr)

        # frames.append(ir_rgb_arr) 

        delay = time.monotonic()-t
        print(f"Loop FPS {int(1/delay)}")

    # TODO: save images in a separate thread worker
    # dump the frames
    # for idx, frame in enumerate(frames):
    #     cv2.imwrite(, frame)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    SIZE = (1024, 768)
    APP_NAME = "Crowd Thermometer"

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
