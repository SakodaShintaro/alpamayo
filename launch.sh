#!/bin/bash

colcon build --packages-select alpamayo_ros
source install/setup.bash
ros2 launch alpamayo_ros alpamayo.launch.py
