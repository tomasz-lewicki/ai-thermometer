# For Jetson inference
import jetson.inference
import jetson.utils

# For ROS core
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# For ROS2 messages
import jsonpickle
from std_msgs.msg import String
from coco_classes import coco_classes

# For display
import cv2

# import scipy.misc
import numpy as np
import copy


class HeatmapClient(Node):
    def __init__(self):
        super().__init__("heatmap_listener")
        self.hm = None
        self.sub = self.create_subscription(String, "heatmap", self.hm_callback, 10)

    def hm_callback(self, msg):
        self.hm = jsonpickle.decode(msg.data)
        print(f"Persons temp. : {round(self.hm.max(), 4)} deg C")


def rgb_to_ir_coords(x_rgb, y_rgb, capture_res=(1280, 720)):

    # Constants
    rgb_native_w, rgb_native_h = 3264, 2464
    ir_w, ir_h = 160, 120
    margin = 0.1
    scale_x = scale_y = rgb_native_w / ir_w * (1 - margin)

    capture_res_w, capture_res_h = capture_res
    scale_x *= capture_res_w / rgb_native_w
    scale_y *= capture_res_h / rgb_native_h

    x_ir = (x_rgb - capture_res_w / 2) / scale_x + ir_w / 2
    y_ir = (y_rgb - capture_res_h / 2) / scale_y + ir_h / 2

    return int(x_ir), int(y_ir)


if __name__ == "__main__":

    # Create detections publisher
    rclpy.init(args=None)
    node = rclpy.create_node("minimal_publisher")
    publisher = node.create_publisher(String, "rgb_detections", 10)

    irclient = HeatmapClient()
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(irclient)

    from threading import Thread

    t = Thread()
    t.run = executor.spin
    t.start()

    # Set up inference
    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)
    camera = jetson.utils.gstCamera(1280, 720, "0")

    # process frames until user exits
    try:
        while True:

            img_ptr, width, height = camera.CaptureRGBA(zeroCopy=1)

            detections = net.Detect(img_ptr, width, height, overlay="none")

            # Copy from cuda memory, so GPU doesn't override the image while visualizing
            # It's only needed if we visualize
            img = copy.copy(jetson.utils.cudaToNumpy(img_ptr, width, height, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # convert to RGB

            for d in detections:

                right, top, left, bottom = (
                    int(d.Right),
                    int(d.Top),
                    int(d.Left),
                    int(d.Bottom),
                )

                top_left, bottom_right = (
                    rgb_to_ir_coords(right, top),
                    rgb_to_ir_coords(bottom, right),
                )

                print("top: ", top_left)
                print("bottom: ", bottom_right)

                print(irclient.hm.max())

                # draw bounding box
                class_name = coco_classes[d.ClassID]
                conf = d.Confidence
                img = cv2.rectangle(img, (right, top), (left, bottom), (255, 0, 0))
                cv2.putText(
                    img,
                    f"{class_name}: {round(conf,4)}",
                    (right, top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                )

            img = cv2.resize(img, (568, 320))
            img = 255 * (img - img.min()) / img.max()  # normalize to 0-255
            img = np.array(img, dtype=np.uint8)  # Convert from fp16 to uint8

            cv2.imshow("Crowd Thermometer Demo", img)
            cv2.waitKey(1)

            # Profiler
            net.PrintProfilerTimes()

            # ROS
            m = String()

            to_send = [
                {"object": d.ClassID, "center_location": d.Center} for d in detections
            ]

            m.data = jsonpickle.encode(to_send)
            publisher.publish(m)
    finally:
        executor.shutdown()
        irclient.destroy_node()
        t.join()
        rclpy.shutdown()
