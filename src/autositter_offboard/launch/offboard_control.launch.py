from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    realsense_dir = get_package_share_directory('realsense2_camera')
    autositter_dir = get_package_share_directory('autositter_offboard')

    params_file = os.path.join(autositter_dir, 'config', 'params.yaml')
    print(params_file)
    print(params_file)
    print(params_file)
    print(params_file)

    return LaunchDescription([
        Node(
            package='autositter_offboard',
            executable='offboard_planner',
            name='offboard_planner',
            output='screen',
            parameters=[params_file]
        ),

        Node(
            package='autositter_offboard',
            executable='flag_detector_node',
            name='flag_detector_node',
            output='screen',
            parameters=[params_file]
        ),

        # Node(
        #     package='autositter_offboard',
        #     executable='webcam_sub',
        #     name='webcam_sub',
        #     output='screen',
        #     parameters=[params_file]
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(realsense_dir, 'launch', 'rs_launch.py')
            ),
            launch_arguments={
                'rgb_camera.color_profile': '960x540x60',
                'rgb_camera.color_format': 'BGR8',
                'enable_depth': 'false'
            }.items()
        )
    ])
