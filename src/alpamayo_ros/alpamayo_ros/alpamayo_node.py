#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that wraps the Alpamayo inference pipeline."""

from __future__ import annotations

import contextlib
import math
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

import rclpy
import torch
from autoware_planning_msgs.msg import Trajectory as PlanningTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import Trigger

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


class AlpamayoRosNode(Node):
    """Minimal ROS 2 interface for the Alpamayo model."""

    def __init__(self) -> None:
        super().__init__("alpamayo_node")

        default_clip = "030c760c-ae38-49aa-9ad8-f5650a545d26"
        default_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name: str = self.declare_parameter("model_name", "nvidia/Alpamayo-R1-10B").value
        self.clip_id_param_name = "clip_id"
        self.declare_parameter(self.clip_id_param_name, default_clip)
        self.declare_parameter("t0_us", 5_100_000)
        self.declare_parameter("maybe_stream", True)
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
        self.declare_parameter("publish_reasoning", True)
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("publisher_queue_size", 10)
        self.declare_parameter("trajectory_time_step", 0.1)

        self._device = torch.device(self.get_parameter("device").value)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warning("CUDA was requested but is not available. Falling back to CPU.")
            self._device = torch.device("cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        queue_size = int(self.get_parameter("publisher_queue_size").value)
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(PlanningTrajectory, traj_topic, queue_size)
        self.get_logger().info(f"Publishing Autoware trajectories on {traj_topic}")

        self._publish_reasoning = bool(self.get_parameter("publish_reasoning").value)
        self._cot_pub = None
        if self._publish_reasoning:
            cot_topic = self.get_parameter("cot_topic").value
            self._cot_pub = self.create_publisher(String, cot_topic, queue_size)
            self.get_logger().info(f"Publishing reasoning traces on {cot_topic}")

        self._trigger_srv = self.create_service(Trigger, "run_inference", self._handle_trigger)

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Future | None = None

        self._model: AlpamayoR1 | None = None
        self._processor = None
        self._model_lock = threading.Lock()

        self._frame_id = self.get_parameter("frame_id").value

        if bool(self.get_parameter("auto_run").value):
            self._submit_inference()

    def destroy_node(self) -> None:
        """Cleanup thread pool before shutting down."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _handle_trigger(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Trigger callback that schedules another inference run."""
        if self._submit_inference():
            response.success = True
            response.message = "Started Alpamayo inference."
        else:
            response.success = False
            response.message = "Inference already running."
        return response

    def _submit_inference(self) -> bool:
        """Submit an inference job if none is running."""
        if self._active_future and not self._active_future.done():
            self.get_logger().info("Inference request ignored because a run is already in progress.")
            return False

        clip_id = str(self.get_parameter(self.clip_id_param_name).value)
        t0_us = int(self.get_parameter("t0_us").value)
        self.get_logger().info(f"Starting inference for clip_id={clip_id} t0_us={t0_us}")

        self._active_future = self._executor.submit(self._run_inference, clip_id, t0_us)
        self._active_future.add_done_callback(lambda future: self._on_future_done(future, clip_id))
        return True

    def _on_future_done(self, future: Future, clip_id: str) -> None:
        """Handle the result of an inference job."""
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo inference failed for clip {clip_id}: {exc}")
            self.get_logger().debug(traceback.format_exc())
            return

        if not metrics:
            return
        msg = (
            f"Alpamayo inference completed for clip {clip_id} "
            f"in {metrics['duration_sec']:.2f}s "
            f"(minADE={metrics['min_ade']:.3f} m, poses={metrics['num_poses']})"
        )
        self.get_logger().info(msg)

    def _ensure_model(self) -> None:
        """Lazily load the Alpamayo model and processor."""
        with self._model_lock:
            if self._model is not None:
                return
            self.get_logger().info(
                f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
            )
            self._model = AlpamayoR1.from_pretrained(self.model_name, dtype=self._dtype).to(
                self._device
            )
            self._model.eval()
            self._processor = helper.get_processor(self._model.tokenizer)

    def _run_inference(self, clip_id: str, t0_us: int) -> dict[str, float] | None:
        """Worker that performs the actual inference."""
        self._ensure_model()

        start = time.time()
        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=t0_us,
            maybe_stream=bool(self.get_parameter("maybe_stream").value),
            num_history_steps=int(self.get_parameter("num_history_steps").value),
            num_future_steps=int(self.get_parameter("num_future_steps").value),
            num_frames=int(self.get_parameter("num_frames").value),
        )

        frames = data["image_frames"].flatten(0, 1)
        messages = helper.create_message(frames)

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
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device=self._device)

        seed = int(self.get_parameter("seed").value)
        torch.manual_seed(seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        num_samples = int(self.get_parameter("num_traj_samples").value)
        num_sets = int(self.get_parameter("num_traj_sets").value)

        generation_kwargs = {
            "top_p": float(self.get_parameter("top_p").value),
            "temperature": float(self.get_parameter("temperature").value),
            "num_traj_samples": num_samples,
            "num_traj_sets": num_sets,
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
        if cot_text and self._cot_pub is not None:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

        min_ade = self._compute_min_ade(pred_xyz_cpu, data["ego_future_xyz"])
        duration = time.time() - start

        return {
            "min_ade": float(min_ade) if min_ade is not None else float("nan"),
            "duration_sec": duration,
            "num_poses": len(traj_msg.points),
        }

    def _to_autoware_trajectory(
        self,
        trajectory: torch.Tensor,
        rotations: torch.Tensor | None,
    ) -> PlanningTrajectory:
        """Convert a sampled trajectory into autoware_planning_msgs/Trajectory."""
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None
        now = self.get_clock().now().to_msg()

        traj_msg = PlanningTrajectory()
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

    def _extract_text(self, extra: dict, key: str) -> str | None:
        """Safely extract a reasoning string from the extra dict."""
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

    def _compute_min_ade(self, pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float | None:
        """Compute minADE across trajectory samples."""
        if gt_xyz is None:
            return None
        try:
            pred_xy = pred_xyz[0, 0, :, :, :2]
            gt_xy = gt_xyz[0, 0, :, :2]
            diff = torch.linalg.norm(pred_xy - gt_xy.unsqueeze(0), dim=-1).mean(dim=-1)
            return float(diff.min().item())
        except Exception:
            self.get_logger().debug("Unable to compute minADE.")
            return None


def main(args: list[str] | None = None) -> None:
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
