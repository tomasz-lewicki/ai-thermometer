# For Jetson inference
import jetson.inference
import jetson.utils

# For ROS core
import time
import rclpy
from rclpy.node import Node

# For ROS2 messages
import jsonpickle
from std_msgs.msg import String
from coco_classes import coco_classes

# For display
import cv2

# import scipy.misc
import numpy as np
import copy

if __name__ == "__main__":

    rclpy.init(args=None)
    node = rclpy.create_node("minimal_publisher")
    publisher = node.create_publisher(String, "rgb_detections", 10)

    # net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.1)
    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)
    camera = jetson.utils.gstCamera(1280, 720, "0")

    # process frames until user exits
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

    node.destroy_node()
    rclpy.shutdown()
