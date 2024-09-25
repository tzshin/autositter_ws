# Basic ROS 2 program to subscribe real-time video stream from realsense2 ROS 2 node.
#
# Minimally modified by Tzu-Hsin TZ Tseng
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv


class WebcamSubscriber(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10)

        self.br = CvBridge()

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')

        current_frame = self.br.imgmsg_to_cv2(data)

        cv.imshow("camera", current_frame)
        cv.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    webcam_subscriber = WebcamSubscriber()

    rclpy.spin(webcam_subscriber)
    webcam_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
