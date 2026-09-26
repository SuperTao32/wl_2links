import time
import math

import mujoco
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("assets/xml/wheel_leg_model.xml")
model = mujoco.MjModel.from_xml_path("assets/xml/wheel_leg_car.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

		# 隔一秒显示一次接触点
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
