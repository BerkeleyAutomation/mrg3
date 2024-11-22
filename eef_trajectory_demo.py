from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
# please use mjpython to run this code on mac

# make sure to pip install mink

from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "kuka_iiwa_14" / "scene.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    # refer to mink for this

    configuration = mink.Configuration(model)
# attachment_site is in xml file for reference
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-2),
    ]



    # inverse k parameters
    
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        rate = RateLimiter(frequency=500.0, warn=False)

        # initialize target
        def generate_random_target():
            # define bounds for random positions (within robot's workspace)
            pos_bounds = {
                'x': (0.3, 0.7),
                'y': (-0.5, 0.5),
                'z': (0.3, 0.8)
            }
            # Generate random position
            pos_target = np.array([
                np.random.uniform(*pos_bounds['x']),
                np.random.uniform(*pos_bounds['y']),
                np.random.uniform(*pos_bounds['z'])
            ])
            # generate random orientation in roll, pitch, yaw (radians)
            rpy_target = np.random.uniform(-np.pi, np.pi, size=3)
            quat_target = R.from_euler('xyz', rpy_target).as_quat()
            #cConvert quaternion from [x, y, z, w] to [w, x, y, z]
            quat_target = np.roll(quat_target, 1)
            # create an SO3 rotation object
            rotation = mink.SO3(wxyz=quat_target)
            # create an SE3 transform with rotation and translation
            T_wt = mink.SE3.from_rotation_and_translation(rotation, pos_target)
            return T_wt

        # set initial target
        T_wt = generate_random_target()
        end_effector_task.set_target(T_wt)

        while viewer.is_running():
            # Compute velocity and integrate into the next configuration.
            for _ in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            
            
            #while loop to generate new random target and move there
            if True:

                T_wt = generate_random_target()
                end_effector_task.set_target(T_wt)
