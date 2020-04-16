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


def rgb_to_ir_coords(y_rgb_px, x_rgb_px, capture_res=(1280, 720)):

    # Constants
    rgb_native_w, rgb_native_h = 3264, 2464
    ir_w, ir_h = 160, 120
    # get coords in (0-1.0) range
    y_rgb = y_rgb_px/capture_res[1]
    x_rgb = x_rgb_px/capture_res[0]

    # x_ir = (x_rgb+0.05) * ir_w * 0.9 
    x_ir = x_rgb * ir_w 
    y_ir = y_rgb * ir_h * 0.75 + 15 

    return [int(y_ir), int(x_ir)]

def ctof(deg_c):
    return round(1.8*deg_c + 32, 2)

if __name__ == "__main__":

    DEMO_HEIGHT = 320
    RGB_SHAPE = (1280, 720)
    # RGB_SHAPE = (3264, 2464)
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
    it = 0
    try:
        while True:
            it +=1
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
                tlc = rgb_to_ir_coords(top, left, capture_res=RGB_SHAPE)  # tlc - top left corner
                brc = rgb_to_ir_coords(bottom, right, capture_res=RGB_SHAPE)  # brc - bottom right corner
                
                ir_w, ir_h = 160, 120  # dimensions of IR sensor

                # clip y coords between 0-119
                tlc[0] = min(max(0, tlc[0]), ir_h - 1)
                brc[0] = min(max(0, brc[0]), ir_h - 1)

                # clip x coords between 0-159
                tlc[1] = min(max(0, tlc[1]), ir_w - 1)
                brc[1] = min(max(0, brc[1]), ir_w - 1)

                obj_heatmap = irclient.hm[tlc[0] : brc[0], tlc[1] : brc[1]]

                if irclient.hm is not None:
                    print(f"Average temperature: {irclient.hm.mean()}")
                    # print(obj_heatmap.max())

                class_name = coco_classes[d.ClassID]
                conf = d.Confidence

                # draw RGB bounding box
                img = cv2.rectangle(img, (right, top), (left, bottom), color=(0, 255, 0), thickness=5)
                
                ambient_temp = irclient.hm.mean()
                temp = obj_heatmap.max() if obj_heatmap.mean() > ambient_temp else obj_heatmap.min()

                cv2.putText(
                    img,
                    text= f"{class_name} ({round(100*conf,2)}%): {temp} deg C {ctof(temp)} deg F",
                    org=(left+10, max(top + 40, 0)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=(255, 255, 255),
                    thickness=2
                )

                # draw IR bounding box
                # img_ir = cv2.rectangle(
                #     img_ir, (tlc[1], tlc[0]), (brc[1], brc[0]), (255, 255, 255)
                # )

            img = img.astype(np.uint8)  # Convert from fp16 to uint8
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.resize(img, (568, 320))

            img_ir = img_ir[15:106 , :]
            # img_ir = cv2.resize(img_ir, (int(4 / 3 * DEMO_HEIGHT), DEMO_HEIGHT))
            img_ir = cv2.applyColorMap(img_ir, cv2.COLORMAP_JET)
            img_ir = cv2.resize(img_ir, (int(16 / 9 * DEMO_HEIGHT), DEMO_HEIGHT))

            # 
            stacked_imgs = np.hstack([img, img_ir])
            
            # display images
            cv2.imwrite(f'/home/nvidia/crowd-thermometer/output/test{it:05}.jpg', stacked_imgs)
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
