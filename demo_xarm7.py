#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron



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


trajectories = np.load("xarm_trajectories.npy")
if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_path("xarm7/scene.xml")
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
        "left_driver_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
         "right_driver_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint"
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    print(dof_ids)
    key_id = model.key("home").id
    desired_positions = np.zeros(len(dof_ids))
    data.qpos[dof_ids] = desired_positions
    mujoco.mj_step(model, data) 
    # step at least once to load model in viewer
    prev_value = 0
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    
    
    while viewer.is_alive:
        for trajectory in trajectories:    
            viewer.render()
            time.sleep(0.005)
            data.qpos[dof_ids] = trajectory
            mujoco.mj_step(model, data)
    viewer.close()
        

