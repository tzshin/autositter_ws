# The PX4 offboard mission planner.
# TODO: More info here

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pyproj import Geod
from collections import Counter
import math
import json
import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, \
    QoSDurabilityPolicy

from autositter_offboard_msgs.msg import (
    ReconStatus,
    FlagReport
)
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
        return f"NED(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


@dataclass
class ReconTarget:
    name: float
    init_at_x: float
    init_at_y: float
    bearing: float

    recon_round: 0

    is_detected: bool = False
    is_tagged: bool = False
    tag_number: Optional[int] = field(default_factory=list)
    guessed_at_x: Optional[float] = field(default_factory=list)
    guessed_at_y: Optional[float] = field(default_factory=list)
    found_at_x: Optional[float] = field(default_factory=list)
    found_at_y: Optional[float] = field(default_factory=list)

    # Internal mapping for compact serialization
    _compact_key_mapping: Dict[str, str] = field(default_factory=lambda: {
        'name': 'n',
        'init_at_x': 'ix',
        'init_at_y': 'iy',
        'bearing': 'b',
        'recon_round': 'rr',
        'is_detected': 'd',
        'is_tagged': 't',
        'tag_number': 'tn',
        'guessed_at_x': 'gx',
        'guessed_at_y': 'gy',
        'found_at_x': 'fx',
        'found_at_y': 'fy',
    }, init=False, repr=False)

    def to_dict(self, keys: Optional[List[str]] = None, compact: bool = False) -> \
            Dict[str, Any]:
        """
        Serialize the ReconTarget to a dictionary following the NED coordinate
        convention.

        Args:
            keys (Optional[List[str]]): List of attribute names to include in the
                                        serialization. If None, all attributes are
                                        included.
            compact (bool): If True, use abbreviated keys for compactness.
            omit_none (bool): If True, omit attributes with None values from the
                              serialization.

        Returns:
            Dict[str, Any]: Serialized dictionary representing the ReconTarget.
        """
        data = asdict(self)

        # Exclude the internal mapping from serialization
        data.pop('_compact_key_mapping', None)

        if keys is not None:
            data = {k: v for k, v in data.items() if k in keys}

        if compact:
            data = {self._compact_key_mapping.get(
                k, k): v for k, v in data.items()}

        return data

    def to_json(self, keys: Optional[List[str]] = None, compact: bool = False) -> str:
        """
        Serialize the ReconTarget to a JSON string following the NED coordinate
        convention.

        Args:
            keys (Optional[List[str]]): List of attribute names to include in the
                                        serialization. If None, all attributes are
                                        included.
            compact (bool): If True, use abbreviated keys for compactness.
            omit_none (bool): If True, omit attributes with None values from the
                              serialization.

        Returns:
            str: Serialized JSON string representing the ReconTarget.
        """
        return json.dumps(
            self.to_dict(keys, compact), separators=(',', ':'), ensure_ascii=False)


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
        self.recon_status_pub = self.create_publisher(
            ReconStatus,
            'recon_status',
            qos_profile
        )

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
        self.flag_report_sub = self.create_subscription(
            FlagReport,
            'flag_report',
            self.flag_report_callback,
            qos_profile
        )

        self.declare_parameter('general.coord_close_thresh')
        self.declare_parameter('general.coord_approach_thresh')
        # FIXME: Implement this feature
        self.declare_parameter('general.arming_t_max')
        self.declare_parameter('goto_speed.recon_travel_h')
        self.declare_parameter('goto_speed.recon_align_h')
        self.declare_parameter('goto_speed.recon_align_v')
        self.declare_parameter('goto_speed.bombing_align_h')
        self.declare_parameter('goto_speed.bombing_align_v')
        self.declare_parameter('takeoff.height')
        self.declare_parameter('recon.height')
        self.declare_parameter('recon.align.ctrl_p')
        self.declare_parameter('recon.align.v_rate')
        self.declare_parameter('recon.align.max_v_dist')
        self.declare_parameter('recon.target_info')
        self.declare_parameter('recon.circle_recipe')  # FIXME: Imp
        self.declare_parameter('bombing.height')
        self.declare_parameter('bombing.align.ctrl_p')
        self.declare_parameter('bombing.align.v_rate')
        self.declare_parameter('bombing.release_altitude')
        self.declare_parameter('bombing.fuselage_open')
        self.declare_parameter('bombing.fuselage_close')
        self.declare_parameter('landing.crit_height')
        self.declare_parameter('landing.crit_t_max')
        self.declare_parameter('landing.brake_l_open')
        self.declare_parameter('landing.brake_r_open')
        self.declare_parameter('landing.brake_l_close')
        self.declare_parameter('landing.brake_r_close')

        self.general_coord_close_thresh = self.get_parameter(
            'general.coord_close_thresh').value
        self.general_coord_approach_thresh = self.get_parameter(
            'general.coord_approach_thresh').value
        self.general_arming_t_max = self.get_parameter(
            'general.arming_t_max').value
        self.goto_speed_recon_travel_h = self.get_parameter(
            'goto_speed.recon_travel_h').value
        self.goto_speed_recon_align_h = self.get_parameter(
            'goto_speed.recon_align_h').value
        self.goto_speed_recon_align_v = self.get_parameter(
            'goto_speed.recon_align_v').value
        self.goto_speed_bombing_align_h = self.get_parameter(
            'goto_speed.bombing_align_h').value
        self.goto_speed_bombing_align_v = self.get_parameter(
            'goto_speed.bombing_align_v').value
        self.takeoff_height = self.get_parameter('takeoff.height').value
        self.recon_height = self.get_parameter('recon.height').value
        self.recon_align_ctrl_p = self.get_parameter(
            'recon.align.ctrl_p').value
        self.recon_align_v_rate = self.get_parameter(
            'recon.align.v_rate').value
        self.recon_align_max_v_dist = self.get_parameter(
            'recon.align.max_v_dist').value
        self.recon_target_info = self.get_parameter(
            'recon.target_info').value
        self.recon_circle_recipe = self.get_parameter(
            'recon.circle_recipe').value
        self.bombing_height = self.get_parameter('bombing.height').value
        self.bombing_align_ctrl_p = self.get_parameter(
            'bombing.align.ctrl_p').value
        self.bombing_align_v_rate = self.get_parameter(
            'bombing.align.v_rate').value
        self.bombing_release_altitude = self.get_parameter(
            'bombing.release_altitude').value
        self.bombing_fuselage_open = self.get_parameter(
            'bombing.fuselage_open').value
        self.bombing_fuselage_close = self.get_parameter(
            'bombing.fuselage_close').value
        self.landing_crit_height = self.get_parameter(
            'landing.crit_height').value
        self.landing_crit_t_max = self.get_parameter(
            'landing.crit_t_max').value
        self.landing_brake_l_open = self.get_parameter(
            'landing.brake_l_open').value
        self.landing_brake_r_open = self.get_parameter(
            'landing.brake_r_open').value
        self.landing_brake_l_close = self.get_parameter(
            'landing.brake_l_close').value
        self.landing_brake_r_close = self.get_parameter(
            'landing.brake_r_close').value

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_local_coord = NEDCoordinate(
            self.vehicle_local_position.x,
            self.vehicle_local_position.y,
            self.vehicle_local_position.z
        )
        self.vehicle_status = VehicleStatus()
        self.flag_report = FlagReport()

        self.never_armed = True
        self.home_coord = None

        self.recon_progress = []
        self.recon_schedule = []
        self.recon_phase = None
        self.recon_cumulated_offset_x = 0.0
        self.recon_cumulated_offset_y = 0.0
        self.recon_cumulated_offset_z = 0.0
        self.recon_found_a_flag = False
        self.recon_detected_tags = []

        self.landing_crit_ts = None

        self.planning_mode = 'IDLE'
        self.planning_mode_prev = self.planning_mode

        self.debug_ts = None
        self.debug_setpoints = []
        self.debug_tags = []
        self.debug_i = 0

        # self.debug_timer = self.create_timer(
        #     0.5, self.debug_log_callback)

        ctrl_loop_period = 0.05
        self.dt = ctrl_loop_period

        self.ctrl_loop_timer = self.create_timer(
            ctrl_loop_period, self.ctrl_loop_callback)

    # def debug_log_callback(self):
    #     self.get_logger().info(self.vehicle_status)
    #     self.get_logger().info(self.planning_mode)

    def are_coordinates_close(
            self,
            coord1: NEDCoordinate,
            coord2: NEDCoordinate,
            threshold: float) -> bool:
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

    def flag_report_callback(self, flag_report):
        self.flag_report = flag_report

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
        self.get_logger().info('Switching to landing mode')

    def set_bombing_fuselage(self, open: bool):
        """"TODO:"""
        if open:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR,
                param1=float(self.bombing_fuselage_open))
        elif not open:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR,
                param1=float(self.bombing_fuselage_close))

    def set_landing_brake(self, open: bool):
        """"TODO:"""
        if open:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR,
                param2=float(self.landing_brake_l_open),
                param3=float(self.landing_brake_r_open))
        elif not open:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR,
                param2=float(self.landing_brake_l_close),
                param3=float(self.landing_brake_r_close))

    def publish_offboard_control_heartbeat(self, enable_offboard: True):
        """
        Publish an OffboardControlMode message with TODO:
        """
        msg = OffboardControlMode(
            position=enable_offboard,
            timestamp=int(self.get_clock().now().nanoseconds / 1000)
        )

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

    def prepare_recon_schedule(self):
        """
        """
        if self.recon_schedule:
            return  # Schedule non_empty, skip planning

        n_detected = sum(1 for rt in self.recon_progress if rt.is_detected)
        n_tagged = sum(1 for rt in self.recon_progress if rt.is_tagged)

        if n_detected == len(self.recon_progrress) \
                and n_tagged == len(self.recon_progress) - 1:
            return  # Targets all detected (with n-1 tagged), skip planning

        if n_detected != len(self.recon_progress):
            # Not all targets are detected
            targets_need_recon = [
                rt for rt in self.recon_progress if not rt.is_detected]
            for rt in targets_need_recon:
                recon_round = rt.recon_round
                if recon_round < len(self.recon_circle_recipe):
                    radius, num_points = self.recon_circle_recipe[recon_round]
                    circle = self.generate_recon_circle(rt, radius, num_points)
                    self.recon_schedule.extend(circle)
                    rt.recon_round += 1
                else:
                    self.get_logger().warn('recon_circle_recipe out of range!')
                    return

        elif n_tagged != len(self.recon_progress) - 1:
            # Some targets require tag re-detection
            targets_need_recon = [
                rt for rt in self.recon_progress if not rt.is_tagged]
            for rt in targets_need_recon:
                copied_rt = copy.deepcopy(rt)
                self.recon_schedule.append(copied_rt)

    def generate_recon_circle(
        self,
        target: ReconTarget,
        radius: float,
        num_points: int
    ) -> List[ReconTarget]:
        """
        Generate points on the circumference of a circle in the NED frame.

        This method creates a list of NEDCoordinate points evenly distributed
        along the circumference of a circle defined by the center and radius.

        Args:
            target (ReconTarget): The target in ReconTarget for use as the center of the
                                  circle.
            radius (float): The radius of the circle in meters.
            num_points (int): The number of points to generate along the circumference.

        Returns:
            List[NEDCoordinate]: A list of NEDCoordinate points on the circle.
        """
        circle = []
        for i in range(num_points):
            new_target = copy.deepcopy(target)

            angle = 2 * math.pi * i / num_points
            new_target.guessed_at_x = target.init_at_x + \
                radius * math.cos(angle)
            new_target.guessed_at_y = target.init_at_y + \
                radius * math.sin(angle)
            new_target.recon_round = target.recon_round + 1

            circle.append(new_target)

        return circle

    def coord_from_recon_target(
            self,
            rt: ReconTarget,
            xy_source: str,
            height: float
    ) -> NEDCoordinate:
        """
        """
        coord = NEDCoordinate()
        match xy_source:
            case 'init':
                coord.x = rt.init_at_x
                coord.y = rt.init_at_y
            case 'guessed':
                coord.x = rt.guessed_at_x
                coord.y = rt.guessed_at_y
            case 'found':
                coord.x = rt.found_at_x
                coord.y = rt.found_at_y
        coord.z = - height

        return coord

    def get_ned_error(self) -> tuple[float, float]:
        img_error_x = self.flag_report.error_x
        img_error_y = self.flag_report.error_y
        body_error_x = - img_error_x  # Camera orientation dependant
        body_error_y = - img_error_y  # Camera orientation dependant
        cos_rot = math.cos(self.vehicle_local_position.heading)
        sin_rot = math.sin(self.vehicle_local_position.heading)
        ned_error_x = body_error_x * cos_rot - body_error_y * sin_rot
        ned_error_y = body_error_x * sin_rot - body_error_y * cos_rot

        return ned_error_x, ned_error_y

    def handle_state_idle(self):
        if self.never_armed \
                and self.vehicle_status.timestamp > 2000 \
                and self.vehicle_status.pre_flight_checks_pass \
                and self.vehicle_status.nav_state \
                == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.planning_mode = 'TAKEOFF'
            self.never_armed = False
            self.home_coord = NEDCoordinate(
                self.vehicle_local_position.x,
                self.vehicle_local_position.y,
                self.vehicle_local_position.z
            )
            # for info in self.recon_target_info: # FIXME:
            #     name = info.get('name', 0.0)
            #     lon_target = info.get('lon', 0.0)
            #     lat_target = info.get('lat', 0.0)
            #     bearing = info.get('bearing', 0.0)
            #     lon_ref = self.vehicle_local_position.ref_lon
            #     lat_ref = self.vehicle_local_position.ref_lat
            #     x_target, y_target = self.global_ll_to_local_xy(
            #         lon_ref, lat_ref, lon_target, lat_target, 0.0, 0.0)
            #     rt = ReconTarget(
            #         name, x_target, y_target, bearing)
            #     self.recon_progress.append(rt)
            self.engage_offboard_mode()
            self.arm()
        else:
            # TODO: Probably streaming vehicle status?
            self.set_bombing_fuselage(open=False)
            self.set_landing_brake(open=True)
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
                self.general_coord_close_thresh):
            self.planning_mode = 'RECON'

            # FIXME: Remove me after debug
            lonlats = [[115.1428860, 38.5595804], [115.1430970, 38.5596965],
                       [115.1429515, 38.5598587], [115.1427404, 38.5596994]]
            xy = []
            for lonlat in lonlats:
                lon, lat = lonlat
                lon_ref = self.vehicle_local_position.ref_lon
                lat_ref = self.vehicle_local_position.ref_lat
                xy.append(self.global_ll_to_local_xy(
                    lon_ref, lat_ref, lon, lat, 0.0, 0.0))
            self.debug_setpoints = [
                NEDCoordinate(
                    self.home_coord.x + xy[0][0],
                    self.home_coord.y + xy[0][1],
                    self.home_coord.z - self.bombing_height),
                NEDCoordinate(
                    self.home_coord.x + xy[1][0],
                    self.home_coord.y + xy[1][1],
                    self.home_coord.z - self.bombing_height),
                NEDCoordinate(
                    self.home_coord.x + xy[2][0],
                    self.home_coord.y + xy[2][1],
                    self.home_coord.z - self.bombing_height),
                NEDCoordinate(
                    self.home_coord.x + xy[3][0],
                    self.home_coord.y + xy[3][1],
                    self.home_coord.z - self.bombing_height),
            ]
        else:
            self.publish_goto_setpoint(takeoff_coord)

    def handle_state_recon(self):
        self.prepare_recon_schedule()
        if self.recon_schedule:
            # Schedule non-empty, preparing the first task...
            fresh_target = self.recon_schedule[0]
            fresh_coord = self.coord_from_recon_target(
                fresh_target, 'guessed', self.recon_height)
        else:
            # Schedule empty, recon finished, let's go bombing
            self.planning_mode = 'BOMBING'  # TODO: Is there's some more thing to do?

        match self.recon_phase:
            case None:
                # Just take off, flying to recon area
                self.publish_goto_setpoint(fresh_coord)
                if self.are_coordinates_close(
                        self.vehicle_local_coord,
                        fresh_coord,
                        self.general_coord_approach_thresh):
                    self.recon_phase = 'travel'

            case 'travel':
                # Travel to the scheduled recon point
                self.publish_goto_setpoint(
                    fresh_coord,
                    max_horizontal_speed=self.goto_speed_recon_travel_h)
                if self.are_coordinates_close(
                        self.vehicle_local_coord,
                        fresh_coord,
                        self.general_coord_approach_thresh):
                    self.recon_phase = 'align'

            case 'align':
                # Find the flag and position the aircraft on top of it
                ned_error_x, ned_error_y = self.get_ned_error()
                delta_x = self.recon_align_ctrl_p * ned_error_x * self.dt
                delta_y = self.recon_align_ctrl_p * ned_error_y * self.dt
                self.recon_cumulated_offset_x += delta_x
                self.recon_cumulated_offset_y += delta_y
                self.recon_cumulated_offset_z += self.recon_align_v_rate * self.dt
                modified_coord = copy.deepcopy(fresh_coord)
                modified_coord.x += self.recon_cumulated_offset_x
                modified_coord.y += self.recon_cumulated_offset_y
                modified_coord.z += self.recon_cumulated_offset_z
                self.publish_goto_setpoint(
                    modified_coord,
                    max_horizontal_speed=self.goto_speed_recon_align_h,
                    max_vertical_speed=self.goto_speed_recon_align_v)
                if self.flag_report.exist_valid_flag:
                    # Detector found a valid flag
                    self.recon_found_a_flag = True
                    self.recon_detected_tags.append(
                        self.flag_report.tag_number)
                elif abs(modified_coord - fresh_coord) \
                        > self.recon_align_max_v_dist:
                    # Align phase terminated, handling results
                    if self.recon_found_a_flag:
                        # Update the results to the corresponding ReconTarget stored in
                        # self.recon_progess
                        for rt in self.recon_progress:
                            if rt.name == fresh_target.name:
                                most_common_tag = Counter(
                                    self.recon_detected_tags).most_common(1)[0][0]
                                rt.is_detected = True
                                rt.is_tagged = True if most_common_tag else False
                                rt.tag_number = most_common_tag
                                rt.guessed_at_x = fresh_coord.x
                                rt.guessed_at_y = fresh_coord.y
                                rt.found_at_x = modified_coord.x
                                rt.found_at_y = modified_coord.y
                        self.recon_schedule = [rt for rt in self.recon_schedule if
                                               rt.name != fresh_target.name]
                    else:
                        self.recon_schedule.pop()

                    self.recon_cumulated_offset_x = 0.0
                    self.recon_cumulated_offset_y = 0.0
                    self.recon_cumulated_offset_z = 0.0
                    self.recon_found_a_flag = False
                    self.recon_detected_tags = []

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
                self.general_coord_close_thresh):
            self.planning_mode = 'LANDING'
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
        if self.vehicle_local_coord.z > landing_crit_coord.z:
            if self.landing_crit_ts is None:
                self.landing_crit_ts = self.vehicle_status.timestamp
            # elif self.vehicle_status.timestamp - self.landing_crit_ts \
            #         > self.landing_crit_t_max \
            #         and self.vehicle_status.arming_state \
            #         != VehicleStatus.ARMING_STATE_DISARMED:
            elif self.vehicle_status.timestamp - self.landing_crit_ts \
                    > self.landing_crit_t_max * 1000.0:
                self.disarm(force=True)
                self.set_landing_brake(open=False)
                # self.set_landing_brake(open=False) # FIXME: Test thiss

        # TODO: Streaming and saving recon result here

    def ctrl_loop_callback(self):
        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_offboard_control_heartbeat(enable_offboard=True)
        else:
            self.publish_offboard_control_heartbeat(enable_offboard=False)

        match self.planning_mode:
            case 'IDLE':
                self.handle_state_idle()
            case 'TAKEOFF':
                self.handle_state_takeoff()
            case 'RECON':
                # FIXME: Remove me after debug
                self.publish_goto_setpoint(
                    self.debug_setpoints[self.debug_i]
                )
                if self.flag_report.exist_valid_flag \
                        and len(self.debug_tags) <= self.debug_i:
                    self.debug_tags.append(self.flag_report.tag_number)
                if self.are_coordinates_close(
                        self.vehicle_local_coord,
                        self.debug_setpoints[self.debug_i],
                        self.general_coord_close_thresh):
                    if len(self.debug_tags) < self.debug_i:
                        self.debug_tags.append(None)
                    self.debug_i += 1
                if self.debug_i == len(self.debug_setpoints):
                    def second_largest_index(lst):
                        valid_numbers = [x for x in lst if x is not None]
                        sorted_unique = sorted(
                            set(valid_numbers), reverse=True)
                        second_largest = sorted_unique[1]
                        return lst.index(second_largest)

                    bomb_idx = second_largest_index(self.debug_tags)
                    self.publish_goto_setpoint(self.debug_setpoints[bomb_idx])
                    if self.are_coordinates_close(
                            self.vehicle_local_coord,
                            self.debug_setpoints[bomb_idx],
                            self.general_coord_close_thresh):
                        # go bombing
                        self.set_bombing_fuselage(open=True)
                        self.planning_mode = 'RETURN'
                        pass

                # self.handle_state_recon()
                # case 'BOMBING':
                # self.handle_state_bombing()
            case 'RETURN':
                self.handle_state_return()
            case 'LANDING':
                self.handle_state_landing()
                self.get_logger().info(self.debug_tags)

        if self.vehicle_status.armed_time >= self.general_arming_t_max * 1000.0:
            self.planning_mode = 'RETURN'

        # self.get_logger().info(f'{self.vehicle_status}')

        if (self.planning_mode_prev != self.planning_mode):
            self.planning_mode_prev = self.planning_mode
            self.get_logger().info(self.planning_mode)
            self.get_logger().info(f'{self.vehicle_status.timestamp}')


def main(args=None):
    rclpy.init(args=args)

    planner = OffboardPlanner()

    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
