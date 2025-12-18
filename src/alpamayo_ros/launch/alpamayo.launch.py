from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch Alpamayo node that listens to live camera + odometry topics."""
    default_camera_topics = [
        "/sensing/camera/front_wide/image",
        "/sensing/camera/front_tele/image",
        "/sensing/camera/cross_left/image",
        "/sensing/camera/cross_right/image",
    ]
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
                        "camera_topics": default_camera_topics,
                        "odometry_topic": "/localization/kinematic_state",
                        "trajectory_topic": "/alpamayo/predicted_trajectory",
                        "cot_topic": "/alpamayo/reasoning",
                        "inference_period_sec": 1.0,
                    }
                ],
            )
        ]
    )
