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
    margin = 0.15
    scale_x = rgb_native_w / ir_w * (1 - margin)
    scale_y = rgb_native_h / ir_h * (1 - margin)

    capture_res_w, capture_res_h = capture_res
    scale_x *= capture_res_w / rgb_native_w
    scale_y *= capture_res_h / rgb_native_h

    x_ir = (x_rgb - capture_res_w / 2) / scale_x + ir_w / 2
    y_ir = (y_rgb - capture_res_h / 2) / scale_y + ir_h / 2

    return [int(x_ir), int(y_ir)]


if __name__ == "__main__":

    DEMO_HEIGHT = 320
    RGB_SHAPE = (1920, 1080)

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
    camera = jetson.utils.gstCamera(*RGB_SHAPE, "0")

    # process frames until user exits
    try:
        while True:

            img_ptr, width, height = camera.CaptureRGBA(zeroCopy=1)

            detections = net.Detect(img_ptr, width, height, overlay="none")

            # Copy from cuda memory, so GPU doesn't override the image while visualizing
            # It's only needed if we visualize
            img = copy.copy(jetson.utils.cudaToNumpy(img_ptr, width, height, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # convert to RGB

            # Get an image from IR listener (as a copy, because we will draw over it)
            img_ir = copy.copy(irclient.hm)
            img_ir = cv2.normalize(img_ir, None, 0, 255, cv2.NORM_MINMAX)
            img_ir = img_ir.astype(np.uint8)
            img_ir = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2BGR)

            for d in detections:

                right, top, left, bottom = (
                    int(d.Right),
                    int(d.Top),
                    int(d.Left),
                    int(d.Bottom),
                )

                # transform coordinates to IR sensor
                tlc = rgb_to_ir_coords(left, top, capture_res=RGB_SHAPE)  # tlc - top left corner
                brc = rgb_to_ir_coords(right, bottom, capture_res=RGB_SHAPE)  # brc - bottom right corner

                # clip x coords between 0-159
                ir_w, ir_h = 160, 120  # dimensions of IR sensor
                tlc[0] = min(max(0, tlc[0]), ir_w - 1)
                brc[0] = min(max(0, brc[0]), ir_w - 1)

                # clip y coords between 0-119
                tlc[1] = min(max(0, tlc[1]), ir_h - 1)
                brc[1] = min(max(0, brc[1]), ir_h - 1)

                obj_heatmap = irclient.hm[tlc[0] : brc[0], tlc[1] : brc[1]]

                if irclient.hm is not None:
                    print(f"Average temperature: {irclient.hm.mean()}")
                    # print(obj_heatmap.max())

                class_name = coco_classes[d.ClassID]
                conf = d.Confidence

                # draw RGB bounding box
                img = cv2.rectangle(img, (right, top), (left, bottom), (255, 0, 0))

                cv2.putText(
                    img,
                    text=f"{class_name} ({round(100*conf,2)}%): {round(obj_heatmap.max(),1)} deg C",
                    org=(left, max(top - 20, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=3
                )

                # draw IR bounding box
                img_ir = cv2.rectangle(
                    img_ir, (tlc[0], tlc[1]), (brc[0], brc[1]), (255, 255, 255)
                )

            img = img.astype(np.uint8)  # Convert from fp16 to uint8
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.resize(img, (568, 320))

            img_ir = cv2.resize(img_ir, (int(4 / 3 * DEMO_HEIGHT), DEMO_HEIGHT))

            # display
            stacked_imgs = np.hstack([img, img_ir])
            cv2.imshow("Heatmap", stacked_imgs)
            cv2.waitKey(1)

            # Profiler
            net.PrintProfilerTimes()

            # Publish detections in a ROS topic
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
