import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Whether to use simulation time'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz2'
    )

    parent_frame_arg = DeclareLaunchArgument(
        'parent_frame',
        default_value='world',
        description='Parent frame for the landmark static transform'
    )

    offset_x_arg = DeclareLaunchArgument(
        'offset_x',
        default_value='0.0',
        description='X offset for the landmark'
    )

    offset_y_arg = DeclareLaunchArgument(
        'offset_y',
        default_value='0.0',
        description='Y offset for the landmark'
    )

    offset_z_arg = DeclareLaunchArgument(
        'offset_z',
        default_value='0.0',
        description='Z offset for the landmark'
    )

    fix_origin_x_arg = DeclareLaunchArgument(
        'fix_origin_x',
        default_value=True,
        description='X-axis of mid point of both shoulder as origin'
    )

    fix_origin_y_arg = DeclareLaunchArgument(
        'fix_origin_y',
        default_value=True,
        description='Y-axis of mid point of both shoulder as origin'
    )

    fix_origin_z_arg = DeclareLaunchArgument(
        'fix_offset_z',
        default_value=False,
        description='Z-axis of mid point of both shoulder as origin'
    )

    use_rviz = LaunchConfiguration('use_rviz')
    parent_frame = LaunchConfiguration('parent_frame')
    offset_x = LaunchConfiguration('offset_x')
    offset_y = LaunchConfiguration('offset_y')
    offset_z = LaunchConfiguration('offset_z')
    
    fix_origin_x = LaunchConfiguration('fix_origin_x')
    fix_origin_y = LaunchConfiguration('fix_origin_y')
    fix_origin_z = LaunchConfiguration('fix_origin_z')

    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rosbridge_server'),
                'launch',
                'rosbridge_websocket_launch.xml'
            )
        )
    )

    processor_node = Node(
        package='robot_teleop',
        executable='landmark_processor',
        name='landmark_processor',
        output='screen',
        parameters=[
            {'fix_x': fix_origin_x},
            {'fix_y': fix_origin_y},
            {'fix_z': fix_origin_z},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    marker_landmark_node = Node(
        package='robot_teleop',
        executable='landmark_marker',
        name='landmark_marker',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='landmark_static_tf',
        output='screen',
        arguments=[
            '--x', offset_x,
            '--y', offset_y,
            '--z', offset_z,
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '3.14159265',
            '--frame-id', parent_frame,
            '--child-frame-id', 'landmark'
        ],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    rviz_config_dir = os.path.join(
        get_package_share_directory('robot_teleop'),
        'rviz',
        'hand_tracking.rviz' 
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen',
        condition=IfCondition(use_rviz),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        use_sim_time,
        use_rviz_arg,
        parent_frame_arg,
        offset_x_arg,
        offset_y_arg,
        offset_z_arg,
        fix_origin_x_arg,
        fix_origin_y_arg,
        fix_origin_z_arg,
        rosbridge_launch,
        processor_node,
        marker_landmark_node,
        static_tf_node,
        rviz_node
    ])