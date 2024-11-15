import argparse

import numpy as np
import time 


import numpy as np
try:
    import mujoco_viewer
except ImportError as e:
    raise ImportError(
        "MuJoCo viewer not found, " "try ``pip install mujoco-python-viewer``"
    ) from e


import mujoco


trajectories = np.load("ur5trajectories.npy")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="name of the robot description")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    mujoco.mj_step(model, data) 
    # join names we are going to control
    joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint"
]

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    print(dof_ids)
    desired_positions = np.zeros(len(dof_ids))
    data.qpos[dof_ids] = desired_positions
    mujoco.mj_step(model, data) 
    # # step at least once to load model in viewer
    while viewer.is_alive:
        for trajectory in trajectories:    
        
            time.sleep(0.005)
            data.qpos[dof_ids] = trajectory
            mujoco.mj_step(model, data)
            viewer.render()
    viewer.close()
        

