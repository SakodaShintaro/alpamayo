#!/bin/bash
set -eux

ros2 bag play ./local/input_bag --clock 200 --rate 0.1
