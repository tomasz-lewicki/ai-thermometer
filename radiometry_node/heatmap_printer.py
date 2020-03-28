import rclpy
from rclpy.node import Node

from std_msgs.msg import String
import jsonpickle
import cv2

class Listener(Node):

    def __init__(self):
        super().__init__('heatmap_printer')
        self.sub = self.create_subscription(String, 'heatmap', self.chatter_callback, 10)

    def chatter_callback(self, msg):
        hmap = jsonpickle.decode(msg.data)
        print(hmap)
        # cv2.imshow('heatmap', hmap)


# The simplest listener for the heatmap:
# Subscribe, and print out the mean of each heatmap

# a.k.a World's Most Complicated Room Thermometer

if __name__ == '__main__':
    rclpy.init(args=None)

    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()