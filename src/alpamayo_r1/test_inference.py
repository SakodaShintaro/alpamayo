# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import time

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
base_model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

# The tokenizer output gets mutated during sampling so keep a primed copy around.
base_model_inputs = helper.to_device(base_model_inputs, "cuda")
primed_inputs = {
    key: value.detach().clone()
    if torch.is_tensor(value)
    else value
    for key, value in base_model_inputs["tokenized_data"].items()
}


def clone_model_inputs(input_dict):
    """Return a fresh copy of the model inputs for stateless inference calls."""

    cloned = dict(input_dict)
    cloned["tokenized_data"] = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in primed_inputs.items()
    }
    return cloned


def run_inference(curr_model, curr_inputs, *, return_extra=False):
    """Runs a single inference pass and optionally returns the extra metadata."""

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return curr_model.sample_trajectories_from_data_with_vlm_rollout(
                data=curr_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=return_extra,
            )


torch.cuda.manual_seed_all(42)
pred_xyz, pred_rot, extra = run_inference(model, clone_model_inputs(base_model_inputs), return_extra=True)

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)

num_timing_runs = 100
print(f"\nMeasuring inference latency over {num_timing_runs} runs...")
durations = []
for _ in range(num_timing_runs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    run_inference(model, clone_model_inputs(base_model_inputs), return_extra=False)
    torch.cuda.synchronize()
    durations.append(time.perf_counter() - start)

sorted_durations = np.sort(np.array(durations))
trim = int(num_timing_runs * 0.1)
if trim > 0 and len(sorted_durations) > 2 * trim:
    central = sorted_durations[trim:-trim]
else:
    central = sorted_durations

central_mean = float(np.mean(central))
central_variance = float(np.var(central))

print(
    "Inference latency (central 80% of runs):",
    f"mean = {central_mean:.6f} seconds, variance = {central_variance:.6f} seconds^2",
)
