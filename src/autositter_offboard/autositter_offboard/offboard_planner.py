# The PX4 offboard mission planner.
# TODO: More info here

from dataclasses import dataclass
from typing import Optional, List
from pyproj import Geod
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, \
    QoSDurabilityPolicy

from autositter_offboard_msgs.msg import FlagReport
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleRatesSetpoint,
    GotoSetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus)


@dataclass
class NEDCoordinate:
    """
    Represents a position in the North, East, Down (NED) coordinate frame.

    Attributes:
        x (float): North position in meters.
        y (float): East position in meters.
        z (float): Down position in meters.
    """
    x: float
    y: float
    z: float

    def __str__(self):
        return f"NED(x={self.x}, y={self.y}, z={self.z})"


@dataclass
class ReconReport:
    well_id: int
    coor: NEDCoordinate
    flag_report: FlagReport


class OffboardPlanner(Node):
    def __init__(self):
        super().__init__('offboard_planner')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile)
        self.vehicle_rates_setpoint_pub = self.create_publisher(
            VehicleRatesSetpoint,
            '/fmu/in/vehicle_rates_setpoint',
            qos_profile)
        self.goto_setpoint_pub = self.create_publisher(
            GotoSetpoint,
            '/fmu/in/goto_setpoint',
            qos_profile)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile)

        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_local_coord = NEDCoordinate(
            self.vehicle_local_position.x,
            self.vehicle_local_position.y,
            self.vehicle_local_position.z
        )
        self.vehicle_status = VehicleStatus()

        self.declare_parameter('bomber_servo_open')
        self.declare_parameter('bomber_servo_close')
        self.declare_parameter('takeoff_height')
        self.declare_parameter('landing_crit_height', 1.0)
        self.declare_parameter('landing_crit_t_max', 2.0)
        self.declare_parameter('coord_close_thresh', 0.3)

        self.bomber_servo_open = self.get_parameter('bomber_servo_open').value
        self.bomber_servo_close = self.get_parameter(
            'bomber_servo_close').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.landing_crit_height = self.get_parameter(
            'landing_crit_height').value
        self.landing_crit_t_max = self.get_parameter(
            'landing_crit_t_max').value
        self.coord_close_thresh = self.get_parameter(
            'coord_close_thresh').value

        self.never_armed = True
        self.home_coord = None
        self.landing_crit_ts = None

        self.current_state = 'IDLE'
        self.last_state = self.current_state

        ctrl_loop_period = 0.05

        self.timer = self.create_timer(
            ctrl_loop_period, self.ctrl_loop_callback)

    def are_coordinates_close(
            coord1: NEDCoordinate, coord2: NEDCoordinate, threshold: float) -> bool:
        """
        Determine if two NED coordinates are within a specified distance of each other.

        Args:
            coord1 (NEDCoordinate): The first NED coordinate.
            coord2 (NEDCoordinate): The second NED coordinate.
            threshold (float): The distance threshold in meters.

        Returns:
            bool: True if the distance between coord1 and coord2 is less than or equal
                  to threshold, False otherwise.
        """
        delta_x = coord2.x - coord1.x
        delta_y = coord2.y - coord1.y
        delta_z = coord2.z - coord1.z

        distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        return distance <= threshold

    def vehicle_local_position_callback(self, vehicle_local_position):
        self.vehicle_local_position = vehicle_local_position
        self.vehicle_local_coord.x = self.vehicle_local_position.x
        self.vehicle_local_coord.y = self.vehicle_local_position.y
        self.vehicle_local_coord.z = self.vehicle_local_position.z

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the flight stack."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0)
        self.get_logger().info('Sending arm command')

    def disarm(self, force=False):
        """Send a disarm command to the flight stack."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
            param2=0.0 if not force else 21196.0)
        self.get_logger().info('Sending disarm command')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0)
        self.get_logger().info('Switching to offboard mode')

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Switching to land mode')

    def drop_bomb(self):
        # TODO: Docstring here
        # self.publish_vehicle_command(
        #     VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR,
        #     param1=float(self.temp_servo_value))
        pass

    def publish_offboard_control_heartbeat(self, control_mode: str):
        """
        Publish an OffboardControlMode message with the specified control mode active.

        Args:
            control_mode (str): The control mode to activate.
                Expected values:
                    - 'position'
                    - 'velocity'
                    - 'acceleration'
                    - 'attitude'
                    - 'body_rate'
        """
        msg = OffboardControlMode(
            position=False,
            velocity=False,
            acceleration=False,
            attitude=False,
            body_rate=False,
            timestamp=int(self.get_clock().now().nanoseconds / 1000)
        )

        valid_modes = ['position', 'velocity',
                       'acceleration', 'attitude', 'body_rate']

        if control_mode in valid_modes:
            setattr(msg, control_mode, True)
            self.get_logger().info(f"Control mode set to: {control_mode}")
        else:
            self.get_logger().warn(
                f"Invalid control mode: {control_mode}. No changes made.")

        self.offboard_control_mode_pub.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish a trajectory setpoint. A basic offboard control setpoint type."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)
        self.get_logger().info(f'Publishing position setpoints {[x, y, z]}')

    def publish_body_rate_setpoint(
            self, roll: float, pitch: float, yaw: float, throttle: float):
        """Publish a body rate setpoint. A basic offboard control setpoint type."""
        msg = VehicleRatesSetpoint()
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.thrust_body = [0.0, 0.0, throttle]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_rates_setpoint_pub.publish(msg)
        self.get_logger().info(
            f'Publishing body rate setpoints {[roll, pitch, yaw, throttle]}')

    def publish_goto_setpoint(
        self,
        coord: NEDCoordinate,
        heading: Optional[float] = None,
        max_horizontal_speed: Optional[float] = None,
        max_vertical_speed: Optional[float] = None,
        max_heading_rate: Optional[float] = None
    ):
        """
        Publish a goto setpoint for the vehicle.

        This method publishes a VehicleLocalPositionSetpoint message to guide the
        vehicle to the specified NED coordinates with optional heading and speed
        constraints.

        Args:
            coord (NEDCoordinate): The target position in the NED frame.
            heading (Optional[float]): Desired yaw angle in radians [-pi, pi]. If
                                       provided, heading control is enabled.
            max_horizontal_speed (Optional[float]): Maximum horizontal speed in m/s.
                                                    If provided, sets a non-default
                                                    limit.
            max_vertical_speed (Optional[float]): Maximum vertical speed in m/s.
                                                  If provided, sets a non-default limit.
            max_heading_rate (Optional[float]): Maximum heading rate in rad/s.
                                                If provided, sets a non-default limit.
        """
        msg = GotoSetpoint()
        msg.position = [coord.x, coord.y, coord.z]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        if heading is not None:
            msg.flag_control_heading = True
            msg.heading = heading
        else:
            msg.flag_control_heading = False
            msg.heading = 0.0

        if max_horizontal_speed is not None:
            msg.flag_set_max_horizontal_speed = True
            msg.max_horizontal_speed = max_horizontal_speed
        else:
            msg.flag_set_max_horizontal_speed = False
            msg.max_horizontal_speed = 0.0

        if max_vertical_speed is not None:
            msg.flag_set_max_vertical_speed = True
            msg.max_vertical_speed = max_vertical_speed
        else:
            msg.flag_set_max_vertical_speed = False
            msg.max_vertical_speed = 0.0

        if max_heading_rate is not None:
            msg.flag_set_max_heading_rate = True
            msg.max_heading_rate = max_heading_rate
        else:
            msg.flag_set_max_heading_rate = False
            msg.max_heading_rate = 0.0

        self.goto_setpoint_pub.publish(msg)
        self.get_logger().info(f"Publishing goto setpoint: {coord}")
        if heading is not None:
            self.get_logger().info(f"With heading: {heading} rad")
        if max_horizontal_speed is not None:
            self.get_logger().info(
                f"With max horizontal speed: {max_horizontal_speed} m/s")
        if max_vertical_speed is not None:
            self.get_logger().info(
                f"With max vertical speed: {max_vertical_speed} m/s")
        if max_heading_rate is not None:
            self.get_logger().info(
                f"With max heading rate: {max_heading_rate} rad/s")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get('param1', 0.0)
        msg.param2 = params.get('param2', 0.0)
        msg.param3 = params.get('param3', 0.0)
        msg.param4 = params.get('param4', 0.0)
        msg.param5 = params.get('param5', 0.0)
        msg.param6 = params.get('param6', 0.0)
        msg.param7 = params.get('param7', 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def global_ll_to_local_xy(
        self,
        lon1: float, lat1: float,
        lon2: float, lat2: float,
        x1: float, y1: float
    ) -> tuple[float, float]:
        """
        Convert global WGS-84 coordinates to local NED (North, East) coordinates.

        This function calculates the forward azimuth and distance between two
        geographic points specified by their longitude and latitude. It then
        computes the change in local coordinates (delta_x, delta_y) based on
        the bearing and distance, and updates the initial local position (x1, y1).

        Parameters:
            lon1 (float): Longitude of the reference point in degrees.
            lat1 (float): Latitude of the reference point in degrees.
            lon2 (float): Longitude of the target point in degrees.
            lat2 (float): Latitude of the target point in degrees.
            x1 (float): Initial local X position (North) in meters.
            y1 (float): Initial local Y position (East) in meters.

        Returns:
            tuple[float, float]: Updated local X & Y positions (North, East) in meters.
        """
        geodesic = Geod(ellps='WGS84')
        fwd_azimuth, _, distance = geodesic.inv(lon1, lat1, lon2, lat2)
        azimuth_rad = math.radians(fwd_azimuth)

        delta_x = distance * math.cos(azimuth_rad)
        delta_y = distance * math.sin(azimuth_rad)
        x2 = x1 + delta_x
        y2 = y1 + delta_y

        return x2, y2

    def generate_recon_circle(
        self,
        center: NEDCoordinate,
        radius: float,
        num_points: int
    ) -> List[NEDCoordinate]:
        """
        Generate points on the circumference of a circle in the NED frame.

        This method creates a list of NEDCoordinate points evenly distributed
        along the circumference of a circle defined by the center and radius.

        Args:
            center (NEDCoordinate): The center of the circle in NED coordinates.
            radius (float): The radius of the circle in meters.
            num_points (int): The number of points to generate along the circumference.

        Returns:
            List[NEDCoordinate]: A list of NEDCoordinate points on the circle.
        """
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z
            points.append(NEDCoordinate(x, y, z))

        return points

    def bake_recon_setpoints(self, point: NEDCoordinate):
        pass

    def handle_state_idle(self):
        if self.never_armed \
                and self.vehicle_status.timestamp > 2000 \
                and self.vehicle_status.pre_flight_checks_pass \
                and self.vehicle_status.nav_state \
                == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.current_state = 'TAKEOFF'
            self.never_armed = False
            self.home_coord = NEDCoordinate(
                self.vehicle_local_position.x,
                self.vehicle_local_position.y,
                self.vehicle_local_position.z
            )
            self.engage_offboard_mode()
            self.arm()
        else:
            # TODO: Probably streaming vehicle status?
            pass

    def handle_state_takeoff(self):
        takeoff_coord = NEDCoordinate(
            self.home_coord.x,
            self.home_coord.y,
            self.home_coord.z - self.takeoff_height
        )

        if self.are_coordinates_close(
                self.vehicle_local_coord,
                takeoff_coord,
                self.coord_close_thresh):
            self.current_state = 'RECON'
        else:
            self.publish_goto_setpoint(takeoff_coord)

    def handle_state_recon(self):
        pass

    def handle_state_bombing(self):
        pass

    def handle_state_return(self):
        return_coord = NEDCoordinate(
            self.home_coord.x,
            self.home_coord.y,
            self.home_coord.z - self.takeoff_height
        )

        if self.are_coordinates_close(
                self.vehicle_local_coord,
                return_coord,
                self.coord_close_thresh):
            self.current_state = 'LANDING'
        else:
            self.publish_goto_setpoint(return_coord)

    def handle_state_landing(self):
        landing_crit_coord = NEDCoordinate(
            self.home_coord.x,
            self.home_coord.y,
            self.home_coord.z - self.landing_crit_height
        )

        self.land()

        # Force disarm if landing_crit_t is reached and the vehicle is not disarmed,
        # likely due to a failed auto-landing.
        if self.vehicle_local_coord.z < landing_crit_coord.z:
            if self.landing_crit_ts is None:
                self.landing_crit_ts = self.vehicle_status.timestamp
            elif self.vehicle_status.timestamp - self.landing_crit_ts \
                    > self.landing_crit_t_max \
                    and self.vehicle_status.arming_state \
                    != VehicleStatus.ARMING_STATE_DISARMED:
                self.disarm(force=True)

        # TODO: Streaming and saving recon result here

    def ctrl_loop_callback(self):
        self.publish_offboard_control_heartbeat('position')

        match self.current_state:
            case 'IDEL':
                self.handle_state_idle()
            case 'TAKEOFF':
                self.handle_state_takeoff()
            case 'RECON':
                self.handle_state_recon()
            case 'BOMBING':
                self.handle_state_bombing()
            case 'RETURN':
                self.handle_state_return()
            case 'LANDING':
                self.handle_state_landing()

        if (self.last_state != self.current_state):
            self.last_state = self.current_state
            self.get_logger().info(self.current_state)


def main(args=None):
    rclpy.init(args=args)

    planner = OffboardPlanner()

    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
