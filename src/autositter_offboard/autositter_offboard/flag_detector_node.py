# Import the FlagDetector class from FlagDetector_class.py
from autositter_offboard.FlagDetector import FlagDetector
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, \
    QoSDurabilityPolicy

from autositter_offboard_msgs.msg import FlagReport


class FlagDetectorNode(Node):
    def __init__(self):
        super().__init__('flag_detector_node')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Declare parameters without default values
        self.declare_parameter('color_filters.0')
        self.declare_parameter('rot_compensation_mode')
        self.declare_parameter('safe_column_width_ratio')
        self.declare_parameter('reference_image_path')

        # Get parameters
        self.color_filters = self.get_parameter('color_filters.0').value
        self.rot_compensation_mode = self.get_parameter(
            'rot_compensation_mode').value
        self.safe_column_width_ratio = self.get_parameter(
            'safe_column_width_ratio').value
        self.reference_image_path = self.get_parameter(
            'reference_image_path').value

        formated_color_filters = [{
            'lower': self.color_filters[0:3],
            'upper': self.color_filters[3:6]
        }]

        print(formated_color_filters)
        print(formated_color_filters)
        print(formated_color_filters)
        print(formated_color_filters)

        # Log parameters for debugging
        # self.get_logger().info(f"Color Filters: {self.color_filters}")
        # self.get_logger().info(f"Rotation Compensation Mode: {self.rot_compensation_mode}")  # noqa: E501
        # self.get_logger().info(f"Safe Column Width Ratio: {self.safe_column_width_ratio}")  # noqa: E501
        # self.get_logger().info(f"Reference Image Path: {self.reference_image_path}")

        # Initialize the FlagDetector class with the parameters
        self.flag_detector = FlagDetector(
            reference_path=self.reference_image_path,
            hsv_ranges=formated_color_filters,
            safe_column_width_ratio=self.safe_column_width_ratio
        )

        # Set up the publisher and subscriber
        self.flag_report_pub = self.create_publisher(
            FlagReport,
            'flag_report',
            qos_profile)
        self.webcam_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.webcam_callback,
            qos_profile)

        self.br = CvBridge()

    def webcam_callback(self, frame):
        self.get_logger().warn('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        self.current_frame = self.br.imgmsg_to_cv2(
            frame)

        # Process the frame using the FlagDetector class
        processed_frame = self.flag_detector.process_frame(
            self.current_frame, mode=self.rot_compensation_mode)

        # Optionally, display the processed frame (remove if running headless)
        # cv.imshow('Processed Frame', processed_frame)
        # cv.waitKey(1)

        # Create and publish FlagReport message based on detection
        flag_report_msg = FlagReport()

        # flag_report_msg.is_flag = self.flag_detector.is_flag  # Not implemented
        detected_valid_flag = True if self.flag_detector.is_within_safe_column \
            else False

        flag_report_msg.exist_valid_flag = detected_valid_flag

        # Convert error tuple to appropriate message fields
        flag_report_msg.error_x = \
            float(self.flag_detector.error[0]) if detected_valid_flag else 0.0
        flag_report_msg.error_y = \
            float(self.flag_detector.error[1]) if detected_valid_flag else 0.0

        flag_report_msg.tag_number = \
            int(self.flag_detector.detected_num) if detected_valid_flag else 0
        flag_report_msg.aiming = \
            float(self.flag_detector.aiming) if detected_valid_flag else 0.0

        self.flag_report_pub.publish(flag_report_msg)


def main(args=None):
    rclpy.init(args=args)

    flag_dector_node = FlagDetectorNode()

    rclpy.spin(flag_dector_node)
    flag_dector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
