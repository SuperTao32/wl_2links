import argparse
import os

import genesis as gs

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Entity Naming Tutorial")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
            link_frame_size=0.15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/robot/wheel_leg_car_2links/wheel_leg_car_2links.urdf",
            pos=(0, 0, 0.245),
        ),
    )

    # imu = scene.add_sensor(
    #     gs.sensors.IMU(
    #         entity_index=robot.idx,
    #         link_idx_local=robot.get_link("base_link").idx_local,
    #         pos_offset=(0, 0, 0.065),
    #     )
    # )

    scene.build()

    # List all entity names
    print(f"All entity names: {scene.entity_names}")

    joints_name = (
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_wheel_joint",
        "right_wheel_joint",
    )
    motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

    ############ Optional: set control gains ############
    # set positional gains
    robot.set_dofs_kp(
        kp=np.array([20, 20, 20, 20, 10, 10]),
        dofs_idx_local=motors_dof_idx,
    )
    # set velocity gains
    robot.set_dofs_kv(
        kv=np.array([5, 5, 5, 5, 5, 5]),
        dofs_idx_local=motors_dof_idx,
    )
    # set force range for safety
    robot.set_dofs_force_range(
        lower=np.array([-10, -10, -10, -10, -10, -10]),
        upper=np.array([10, 10, 10, 10, 10, 10]),
        dofs_idx_local=motors_dof_idx,
    )
    # Hard reset
    # for i in range(150):
    #     if i < 50:
    #         robot.set_dofs_position(np.array([1, 1, -1.5, -1.5, 0, 0]), motors_dof_idx)
    #     elif i < 100:
    #         robot.set_dofs_position(np.array([-1, -1, 1.5, 1.5, 0, 0]), motors_dof_idx)
    #     else:
    #         robot.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), motors_dof_idx)

    #     scene.step()

    # 打印joint和qpos
    print("robot joints:")
    for i, joint in enumerate(robot.joints):
        print(i, joint)
    print("robot qpos:")
    print(robot.get_qpos())

    # PD control
    horizon = 1250
    for i in range(horizon):
        robot.set_dofs_velocity(
            np.array([0, 0, 0, 0, 0, 0])[4:],
            motors_dof_idx[4:],
        )
        robot.set_dofs_position(
            np.array([np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0, 0])[:4],
            motors_dof_idx[:4],
        )
        # # This is the control force computed based on the given control command
        # # If using force control, it's the same as the given control command
        # print("control force:", robot.get_dofs_control_force(motors_dof_idx))

        # # This is the actual force experienced by the dof
        # print("internal force:", robot.get_dofs_force(motors_dof_idx))
        scene.step()


if __name__ == "__main__":
    main()
