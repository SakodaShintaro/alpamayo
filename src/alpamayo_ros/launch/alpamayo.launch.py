from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo inference node with default parameters."""
    return LaunchDescription(
        [
            Node(
                package="alpamayo_ros",
                executable="alpamayo_node",
                name="alpamayo_node",
                output="screen",
                parameters=[
                    {
                        "auto_run": True,
                        "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
                        "t0_us": 5_100_000,
                        "trajectory_topic": "/alpamayo/predicted_path",
                        "cot_topic": "/alpamayo/reasoning",
                    }
                ],
            )
        ]
    )
