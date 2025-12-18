#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo and publishes Autoware trajectories."""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import rclpy
import torch
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoRosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) to run Alpamayo inference."""

    def __init__(self) -> None:
        super().__init__("alpamayo_node")

        default_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name: str = self.declare_parameter("model_name", "nvidia/Alpamayo-R1-10B").value
        self.declare_parameter("num_history_steps", 16)
        self.declare_parameter("num_future_steps", 64)
        self.declare_parameter("num_frames", 4)
        self.declare_parameter("top_p", 0.98)
        self.declare_parameter("temperature", 0.6)
        self.declare_parameter("max_generation_length", 256)
        self.declare_parameter("num_traj_samples", 1)
        self.declare_parameter("num_traj_sets", 1)
        self.declare_parameter("seed", 42)
        self.declare_parameter("auto_run", True)
        self.declare_parameter("device", default_device)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("publisher_queue_size", 10)
        self.declare_parameter("trajectory_time_step", 0.1)
        self.declare_parameter("inference_period_sec", 1.0)
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("camera_topics", [])

        self._device = torch.device(self.get_parameter("device").value)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warning("CUDA was requested but is not available. Falling back to CPU.")
            self._device = torch.device("cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        self._num_history_steps = int(self.get_parameter("num_history_steps").value)
        self._num_frames = int(self.get_parameter("num_frames").value)
        inference_period = float(self.get_parameter("inference_period_sec").value)

        queue_size = int(self.get_parameter("publisher_queue_size").value)
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, queue_size)
        self.get_logger().info(f"Publishing Autoware trajectories on {traj_topic}")

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, queue_size)
        self.get_logger().info(f"Publishing reasoning traces on {cot_topic}")

        self._trigger_srv = self.create_service(Trigger, "run_inference", self._handle_trigger)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        self._model: Optional[AlpamayoR1] = None
        self._processor = None
        self._model_lock = threading.Lock()

        self._frame_id = self.get_parameter("frame_id").value

        camera_topics = list(self.get_parameter("camera_topics").get_parameter_value().string_array_value)
        if not camera_topics:
            raise ValueError("camera_topics parameter must list at least one image topic.")
        self._camera_topics = camera_topics
        self._camera_buffers: Dict[str, deque] = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }
        self._camera_lock = threading.Lock()

        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage, topic, lambda msg, t=topic: self._handle_image(t, msg), 10
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic}")

        self._odometry_buffer: deque[Odometry] = deque(maxlen=self._num_history_steps * 4)
        self._odometry_lock = threading.Lock()
        odom_topic = self.get_parameter("odometry_topic").value
        self.create_subscription(Odometry, odom_topic, self._handle_odometry, 50)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        self._auto_timer = None
        if bool(self.get_parameter("auto_run").value):
            self._auto_timer = self.create_timer(inference_period, self._timer_callback)

    def destroy_node(self) -> None:
        """Cleanup resources before shutting down."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_inference_payload()
        if payload is None:
            return
        self._launch_inference(payload)

    def _handle_trigger(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._active_future and not self._active_future.done():
            response.success = False
            response.message = "Inference already running."
            return response
        payload = self._prepare_inference_payload()
        if payload is None:
            response.success = False
            response.message = "Insufficient sensor data for Alpamayo input."
            return response
        self._launch_inference(payload)
        response.success = True
        response.message = "Alpamayo inference started."
        return response

    def _handle_image(self, topic: str, msg: CompressedImage) -> None:
        tensor = self._compressed_image_to_tensor(msg)
        if tensor is None:
            return
        with self._camera_lock:
            self._camera_buffers[topic].append((msg.header.stamp, tensor))

    def _handle_odometry(self, msg: Odometry) -> None:
        with self._odometry_lock:
            self._odometry_buffer.append(msg)

    def _compressed_image_to_tensor(self, msg: CompressedImage) -> Optional[torch.Tensor]:
        try:
            import cv2
        except ImportError:
            self.get_logger().error("cv2 is required to decode CompressedImage messages.")
            return None
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning("Failed to decode compressed image.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return tensor

    def _launch_inference(self, payload: dict) -> None:
        self.get_logger().info("Starting Alpamayo inference from streaming data.")
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    def _prepare_inference_payload(self) -> Optional[dict]:
        with self._camera_lock:
            if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
                return None
            camera_tensors: List[torch.Tensor] = []
            for topic in self._camera_topics:
                frames = list(self._camera_buffers[topic])[-self._num_frames:]
                tensors = [frame for _, frame in frames]
                camera_tensors.append(torch.stack(tensors, dim=0))
            image_frames = torch.stack(camera_tensors, dim=0)
        with self._odometry_lock:
            if len(self._odometry_buffer) < self._num_history_steps:
                return None
            odom_history = list(self._odometry_buffer)[-self._num_history_steps:]
        ego_history_xyz, ego_history_rot = self._build_history_tensors(odom_history)
        return {
            "image_frames": image_frames,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

    def _build_history_tensors(self, odom_history: List[Odometry]) -> tuple[torch.Tensor, torch.Tensor]:
        positions = []
        rotations = []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rotations.append(Rotation.from_quat(quat).as_matrix())
        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        centered = positions_np - positions_np[-1]
        history_xyz_local = centered @ t0_rot_inv.T
        history_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)
        ego_history_xyz = torch.from_numpy(history_xyz_local).unsqueeze(0).unsqueeze(0)
        ego_history_rot = torch.from_numpy(history_rot_local).unsqueeze(0).unsqueeze(0)
        return ego_history_xyz, ego_history_rot

    def _ensure_model(self) -> None:
        with self._model_lock:
            if self._model is not None:
                return
            self.get_logger().info(
                f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
            )
            self._model = AlpamayoR1.from_pretrained(self.model_name, dtype=self._dtype).to(self._device)
            self._model.eval()
            self._processor = helper.get_processor(self._model.tokenizer)

    def _run_inference(self, payload: dict) -> dict:
        self._ensure_model()
        start = time.time()
        frames = payload["image_frames"]
        messages = helper.create_message(frames.flatten(0, 1))
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": processor_inputs,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device=self._device)

        seed = int(self.get_parameter("seed").value)
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        generation_kwargs = {
            "top_p": float(self.get_parameter("top_p").value),
            "temperature": float(self.get_parameter("temperature").value),
            "num_traj_samples": int(self.get_parameter("num_traj_samples").value),
            "num_traj_sets": int(self.get_parameter("num_traj_sets").value),
            "max_generation_length": int(self.get_parameter("max_generation_length").value),
            "return_extra": True,
        }

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                **generation_kwargs,
            )

        pred_xyz_cpu = pred_xyz.detach().cpu()
        pred_rot_cpu = pred_rot.detach().cpu()
        trajectory = pred_xyz_cpu[0, 0, 0]
        rotation = pred_rot_cpu[0, 0, 0]
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)

        cot_text = self._extract_text(extra, "cot")
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

        duration = time.time() - start
        return {"duration_sec": duration, "num_poses": len(traj_msg.points)}

    def _to_autoware_trajectory(
        self,
        trajectory: torch.Tensor,
        rotations: torch.Tensor | None,
    ) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None
        now = self.get_clock().now().to_msg()

        traj_msg = Trajectory()
        traj_msg.header.stamp = now
        traj_msg.header.frame_id = self._frame_id

        dt = float(self.get_parameter("trajectory_time_step").value)
        prev_xy = None

        for idx, point in enumerate(traj_np):
            traj_point = TrajectoryPoint()
            traj_point.pose.position.x = float(point[0])
            traj_point.pose.position.y = float(point[1])
            traj_point.pose.position.z = float(point[2])

            if rot_np is not None:
                quat = Rotation.from_matrix(rot_np[idx]).as_quat()
                traj_point.pose.orientation.x = float(quat[0])
                traj_point.pose.orientation.y = float(quat[1])
                traj_point.pose.orientation.z = float(quat[2])
                traj_point.pose.orientation.w = float(quat[3])
            else:
                traj_point.pose.orientation.w = 1.0

            if prev_xy is None:
                speed = 0.0
            else:
                dx = float(point[0] - prev_xy[0])
                dy = float(point[1] - prev_xy[1])
                dist = math.hypot(dx, dy)
                speed = dist / dt if dt > 0 else 0.0

            traj_point.longitudinal_velocity_mps = float(speed)
            traj_point.lateral_velocity_mps = 0.0
            traj_point.acceleration_mps2 = 0.0
            traj_point.heading_rate_rps = 0.0

            seconds_float = idx * dt
            seconds_int = int(seconds_float)
            nanosec = int((seconds_float - seconds_int) * 1e9)
            traj_point.time_from_start = Duration(sec=seconds_int, nanosec=nanosec)

            traj_msg.points.append(traj_point)
            prev_xy = (point[0], point[1])

        return traj_msg

    def _extract_text(self, extra: dict, key: str) -> Optional[str]:
        if not extra or key not in extra:
            return None
        text_array = extra[key]
        try:
            text = text_array[0, 0, 0]
        except Exception:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = str(text).strip()
        return text or None

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo inference failed: {exc}")
            return
        if not metrics:
            return
        self.get_logger().info(
            f"Alpamayo inference completed in {metrics['duration_sec']:.2f}s "
            f"(points={metrics['num_poses']})."
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AlpamayoRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
