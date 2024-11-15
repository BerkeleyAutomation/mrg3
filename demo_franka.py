#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""
Show a robot description, specified from the command line, using MuJoCo.

This example requires MuJoCo, which is installed by ``pip install mujoco``, and
the MuJoCo viewer installed by ``pip install mujoco-python-viewer``.
"""

import argparse

import numpy as np
import time 


import numpy as np


import mujoco

try:
    import mujoco_viewer
except ImportError as e:
    raise ImportError(
        "MuJoCo viewer not found, " "try ``pip install mujoco-python-viewer``"
    ) from e

from robot_descriptions.loaders.mujoco import load_robot_description

trajectories = np.load("franka_trajectories.npy")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="name of the robot description")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path("panda/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    mujoco.mj_step(model, data) 
    # join names we are going to control
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    print(dof_ids)
    key_id = model.key("home").id
    desired_positions = np.zeros(len(dof_ids))
    data.qpos[dof_ids] = desired_positions
    mujoco.mj_step(model, data) 
    # step at least once to load model in viewer
    prev_value = 0
    
    
    while viewer.is_alive:
        for trajectory in trajectories:    
            viewer.render()
            time.sleep(0.005)
            thereshold = 0.05
            trajectory[-1] = trajectory[-1] if trajectory[-1] < thereshold else thereshold
            trajectory[-2] = trajectory[-2] if trajectory[-1] < thereshold else thereshold
            data.qpos[dof_ids] = trajectory
            mujoco.mj_step(model, data)
    viewer.close()
        

