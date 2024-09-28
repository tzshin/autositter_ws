from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
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
        )
    ])
