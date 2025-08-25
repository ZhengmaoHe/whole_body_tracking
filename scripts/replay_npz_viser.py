"""This script demonstrates how to use Viser to visualize and replay motion data from an NPZ file.

.. code-block:: bash

    # Usage
    python replay_npz_viser.py --registry_name dance1_subject1
"""

import argparse
import numpy as np
import torch
import time

import viser
from viser.extras import ViserUrdf
import yourdfpy

import os
from typing import Sequence

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions using Viser.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

def main():
    server = viser.ViserServer()

    # Load URDF
    urdf_path = "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
    urdf = yourdfpy.URDF.load(
        urdf_path,
        load_meshes=True,
        build_scene_graph=True,
        load_collision_meshes=False,
        build_collision_scene_graph=False,
    )

    # Create a parent frame for the robot
    robot_base = server.scene.add_frame("/robot", show_axes=False)

    # Create ViserUrdf instance
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False
    )

    print("ViserUrdf actuated joint names:", viser_urdf.get_actuated_joint_names())

    # Load motion
    motion_file = f"./tmp/{args_cli.registry_name}.npz"
    motion = MotionLoader(
        motion_file,
        [0],
        "cpu",
    )
    sim_dt = 0.02

    # Add grid (optional, similar to urdf_viser.py)
    trimesh_scene = viser_urdf._urdf.scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        # position=(
        #     0.0,
        #     0.0,
        #     trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        # ),
    )
    isaac_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
    viser_index = [isaac_names.index(joint_name) for joint_name in viser_urdf.get_actuated_joint_names()]

    import glob
    motion_files = sorted(glob.glob("./tmp/*.npz"))
    motion_names = [os.path.basename(f) for f in motion_files]
    current_motion_idx = motion_names.index(f"{args_cli.registry_name}.npz") if f"{args_cli.registry_name}.npz" in motion_names else 0
    motion_dropdown = server.gui.add_dropdown("Motion File", motion_names, initial_value=motion_names[current_motion_idx])
    playing = server.gui.add_checkbox("Playing", True)
    timestep_slider = server.gui.add_slider("Timestep", min=0, max=0, step=1, initial_value=0)

    def load_motion(file_path):
        nonlocal motion, viser_index
        motion = MotionLoader(file_path, [0], "cpu")
        timestep_slider.max = motion.time_step_total - 1
        return motion

    motion = load_motion(motion_files[current_motion_idx])
    prev_motion = motion_files[current_motion_idx]

    while True:
        with server.atomic():
            if playing.value:
                timestep_slider.value = (timestep_slider.value + 1) % motion.time_step_total
            time_step = timestep_slider.value

            # Check for motion file change
            selected_motion = motion_dropdown.value
            if selected_motion != os.path.basename(prev_motion):
                prev_motion = motion_files[motion_names.index(selected_motion)]
                motion = load_motion(prev_motion)
                time_step = 0
                timestep_slider.value = 0

            # Update root pose
            root_pos = motion.body_pos_w[time_step, 0, :].numpy()  # body 0 is root
            root_quat = motion.body_quat_w[time_step, 0, :].numpy()
            robot_base.position = root_pos
            robot_base.wxyz = root_quat  # Viser uses wxyz order

            # Update joint configuration
            joint_pos = motion.joint_pos[time_step, viser_index].numpy()  # no env dim
            viser_urdf.update_cfg(joint_pos)

        time.sleep(sim_dt if playing.value else 0.1)  # Slower update when paused

if __name__ == "__main__":
    main()
