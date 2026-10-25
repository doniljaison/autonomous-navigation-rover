from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'device_id':       0,
                'deadzone':        0.05,
                'autorepeat_rate': 20.0,
            }]
        ),
        Node(
            package='demo_nodes_py',
            executable='joy_rover_http',
            name='joy_rover_http',
            parameters=[{
                'rover_url':      'http://10.224.13.53/js',
                'axis_rt':        5,
                'axis_lt':        2,
                'axis_angular':   0,
                'max_speed':      0.45,
                'angular_scale':  0.35,
                'enable_button':  4,
                'require_enable': True,
                'cmd_hz':         20.0,
            }]
        ),
    ])