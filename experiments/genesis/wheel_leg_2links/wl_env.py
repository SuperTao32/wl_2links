import math

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import torch
from tensordict import TensorDict

from curriculum import CurriculumManager
from tracking_gate import DEFAULT_TRACKING_GATE_CFG, merge_tracking_gate_config, smooth_gate
from velocity_estimator import (
    complementary_forward_velocity_update,
    gravity_compensated_forward_acceleration,
    wheel_forward_velocity,
)


class Wl_Env:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg=None,
        show_viewer=False,
    ):
        ########### 确定设备 ###########
        self.device = gs.device
        ########### 保存cfg ###########
        self.cfg = env_cfg
        self.env_cfg = env_cfg  # TODO 这里self的cfg和env_cfg的区别？
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.curriculum_cfg = curriculum_cfg or {"enabled": False, "stages": []}
        self.tracking_gate_cfg = {}
        self._apply_tracking_gate(reward_cfg.get("tracking_gate", DEFAULT_TRACKING_GATE_CFG))

        ########### 定义基本参数 ###########
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_joints = env_cfg["num_joints"]
        self.num_hip_joints = env_cfg["num_hip_joints"]
        self.num_knee_joints = env_cfg["num_knee_joints"]
        self.num_wheels = env_cfg["num_wheels"]
        self.num_commands = command_cfg["num_commands"]

        assert self.num_joints == self.num_hip_joints + self.num_knee_joints
        assert self.num_actions == self.num_joints + self.num_wheels

        # 定义训练参数
        self.dt = 0.02
        self.resample_step = self.env_cfg["resampling_time_s"] / self.dt
        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        # 没有该字段的历史训练配置继续使用仿真真值，避免旧 checkpoint 的
        # 32 维 actor 输入被新代码静默改成另一种语义。
        self.policy_velocity_source = obs_cfg.get("policy_velocity_source", "simulator")
        if self.policy_velocity_source not in {"simulator", "imu_wheel_estimator"}:
            raise ValueError(
                "obs_cfg['policy_velocity_source'] must be 'simulator' or 'imu_wheel_estimator'"
            )
        self.imu_cfg = dict(obs_cfg.get("imu", {}))
        self.velocity_estimator_cfg = dict(obs_cfg.get("velocity_estimator", {}))
        self.wheel_radius = float(self.velocity_estimator_cfg.get("wheel_radius", 0.08))
        self.wheel_velocity_sign = float(self.velocity_estimator_cfg.get("wheel_velocity_sign", 1.0))
        self.gravity_magnitude = float(self.velocity_estimator_cfg.get("gravity_magnitude", 9.81))
        self.wheel_correction_time_constant_s = float(
            self.velocity_estimator_cfg.get("wheel_correction_time_constant_s", 0.5)
        )
        self.max_estimated_velocity = float(self.velocity_estimator_cfg.get("max_abs_velocity", 3.0))
        if self.wheel_radius <= 0.0:
            raise ValueError("velocity_estimator.wheel_radius must be positive")
        if self.wheel_velocity_sign not in {-1.0, 1.0}:
            raise ValueError("velocity_estimator.wheel_velocity_sign must be -1.0 or 1.0")
        if self.gravity_magnitude <= 0.0:
            raise ValueError("velocity_estimator.gravity_magnitude must be positive")
        if self.wheel_correction_time_constant_s < 0.0:
            raise ValueError("velocity_estimator.wheel_correction_time_constant_s cannot be negative")
        if self.max_estimated_velocity <= 0.0:
            raise ValueError("velocity_estimator.max_abs_velocity must be positive")

        ########### 创建 scene ###########
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                realtime_factor=1.0 if show_viewer else None,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        # 添加实体
        self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="assets/robot/wheel_leg_car_2links/wheel_leg_car_2links.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )
        self.imu = None
        if self.policy_velocity_source == "imu_wheel_estimator":
            self.imu = self.scene.add_sensor(
                gs.sensors.IMU(
                    entity_idx=self.robot.idx,
                    link_idx_local=self.robot.get_link(self.imu_cfg.get("link_name", "base_link")).idx_local,
                    pos_offset=tuple(self.imu_cfg.get("pos_offset", (0.0, 0.0, 0.0))),
                    acc_noise=self.imu_cfg.get("acc_noise", 0.0),
                    acc_bias=self.imu_cfg.get("acc_bias", 0.0),
                    acc_random_walk=self.imu_cfg.get("acc_random_walk", 0.0),
                    gyro_noise=self.imu_cfg.get("gyro_noise", 0.0),
                    gyro_bias=self.imu_cfg.get("gyro_bias", 0.0),
                    gyro_random_walk=self.imu_cfg.get("gyro_random_walk", 0.0),
                    delay=self.imu_cfg.get("delay", 0.0),
                    jitter=self.imu_cfg.get("jitter", 0.0),
                )
            )

        # build环境
        self.scene.build(n_envs=num_envs)

        ########### 创建索引 ###########
        self.joints_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.knees_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in ["left_knee", "right_knee"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.wheels_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["wheel_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )

        ########### PD 参数 ###########
        self.joint_kp = _gain_tensor(self.env_cfg["joint_kp"], self.num_joints, "joint_kp")
        self.joint_kd = _gain_tensor(self.env_cfg["joint_kd"], self.num_joints, "joint_kd")
        self.wheel_kd = _gain_tensor(self.env_cfg["wheel_kd"], self.num_wheels, "wheel_kd")
        self.robot.set_dofs_kp(self.joint_kp.tolist(), self.joints_dof_idx)
        self.robot.set_dofs_kv(self.joint_kd.tolist(), self.joints_dof_idx)
        self.robot.set_dofs_kv(self.wheel_kd.tolist(), self.wheels_dof_idx)

        ########### 定义重力方向 ###########
        self.global_gravity_dir = torch.tensor(
            [0.0, 0.0, -1.0],
            dtype=gs.tc_float,
            device=gs.device,
        )

        ########### 初始和默认姿态设置 ###########
        # init是每次环境重置的状态量
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat_inv = gs.inv_quat(self.init_base_quat)
        self.init_projected_gravity_dir = transform_by_quat(self.global_gravity_dir, self.init_base_quat_inv)
        self.init_joint_pos = torch.tensor(
            [self.env_cfg["default_joint_pos"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        init_knee_pos = torch.tensor(
            [self.env_cfg["default_joint_pos"][name] for name in ["left_knee", "right_knee"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_leg_length = self._compute_leg_length(init_knee_pos)
        self.init_leg_angle = self._compute_leg_angle(self.init_joint_pos)

        # 这样遍历保留robot的joint顺序方便reset时一起init
        init_dof_pos_list = []
        for joint in self.robot.joints[1:]:
            if joint.n_qs == 0:
                continue
            init_pos = self.env_cfg["default_joint_pos"].get(joint.name, 0.0)
            init_dof_pos_list.append(init_pos)

        self.init_dof_pos = torch.tensor(init_dof_pos_list, dtype=gs.tc_float, device=gs.device)
        # robot的qpos总共是base的pos+quat共7个加上joint+wheel的qpos
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))

        # default是策略的“零点”，策略输出的是偏移
        self.default_joint_pos = torch.tensor(
            [self.env_cfg["default_joint_pos"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )

        ########### 更新buffers, 用于暂存当前状态，指令、动作、奖励、reset ###########
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.target_joint_pos = torch.empty((self.num_envs, self.num_joints), dtype=gs.tc_float, device=gs.device)
        self.target_wheel_vel = torch.empty((self.num_envs, self.num_wheels), dtype=gs.tc_float, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.reward_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.terminated_buf = torch.empty((self.num_envs,), dtype=torch.bool, device=gs.device)
        self.extras = dict()
        # 状态相关 TODO 是否需要last pos/vel
        self.joint_pos = torch.empty((self.num_envs, self.num_joints), dtype=gs.tc_float, device=gs.device)
        self.joint_vel = torch.empty_like(self.joint_pos)
        self.wheel_vel = torch.empty((self.num_envs, self.num_wheels), dtype=gs.tc_float, device=gs.device)
        self.leg_length = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.leg_angle = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        # base_* 保存仿真真值，供 reward、critic 和诊断使用；actor 使用下方的
        # IMU/轮速估计量，防止策略依赖真机无法直接获得的完美线速度。
        self.imu_lin_acc = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.imu_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.forward_kinematic_acc = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.wheel_forward_vel = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.estimated_base_lin_vel = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.height_gate = torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.attitude_gate = torch.ones_like(self.height_gate)
        self.tracking_gate_raw = torch.ones_like(self.height_gate)
        self.tracking_gate = torch.ones_like(self.height_gate)
        # limits & scales
        # scales代表状态、策略、命令与实际机器人动作和指令之间的缩放
        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        # 保留未乘 dt 的原始奖励权重，课程切换时才能正确重新计算运行时权重。
        self.raw_reward_scales: dict[str, float] = dict(reward_cfg["reward_scales"])
        self.reward_scales: dict[str, float] = {}
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["leg_length"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # 这里两个Tensor分别代表min和max
        self.commands_limit: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_range"],
                self.command_cfg["ang_vel_range"],
                self.command_cfg["min_leg_length_range"],
            )
        )

        # 其他buffers
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        ########### 定义奖励函数（与dt相关）###########
        self.reward_functions, self.episode_sums = dict(), dict()
        self._apply_reward_scales(self.raw_reward_scales)
        self.episode_metric_sums = {
            "height_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "attitude_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "tracking_gate": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "tracking_gate_fully_open": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
            "velocity_estimator_abs_error": torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device),
        }

        ########### 课程学习 ###########
        self.global_step = 0
        self.curriculum = CurriculumManager(self.curriculum_cfg)
        # 调度器只负责“何时切换”；以下处理器负责把配置真正写入运行中的环境。
        self.curriculum.register_target("command_ranges", self._apply_command_ranges)
        self.curriculum.register_target("reward_scales", self._apply_reward_scales)
        self.curriculum.register_target("tracking_gate", self._apply_tracking_gate)
        self.curriculum.register_target("termination_limits", self._apply_termination_limits)
        self.curriculum.register_target("action_limits", self._apply_action_limits)
        if self.curriculum.update(0, force=True):
            print(f"[curriculum] stage={self.curriculum.current_stage_name} step=0")

        ########### 初始reset环境 ###########
        self.reset()

    def reset(self):
        self._reset_idx()
        self._update_observations()
        return self.get_observations()

    def step(self, actions):
        # global_step 是控制步数，与 num_envs 无关；课程阶段按这个计数推进。
        self.global_step += 1
        if self.curriculum.update(self.global_step):
            print(f"[curriculum] stage={self.curriculum.current_stage_name} step={self.global_step}")

        ########### 执行动作 ###########
        hip_joint_actions = torch.clip(
            actions[:, : self.num_hip_joints],
            -self.env_cfg["clip_hip_joint_action"],
            self.env_cfg["clip_hip_joint_action"],
        )
        knee_joint_actions = torch.clip(
            actions[:, self.num_hip_joints : self.num_hip_joints + self.num_knee_joints],
            -self.env_cfg["clip_knee_joint_action"],
            self.env_cfg["clip_knee_joint_action"],
        )
        wheel_actions = torch.clip(
            actions[:, self.num_joints : self.num_joints + self.num_wheels],
            -self.env_cfg["clip_wheel_action"],
            self.env_cfg["clip_wheel_action"],
        )
        self.actions = torch.concatenate((hip_joint_actions, knee_joint_actions, wheel_actions), dim=-1)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_hip_pos = exec_actions[:, : self.num_hip_joints] * self.env_cfg["hip_joint_pos_scale"]
        target_knee_pos = (
            exec_actions[:, self.num_hip_joints : self.num_hip_joints + self.num_knee_joints]
            * self.env_cfg["knee_joint_pos_scale"]
        )
        target_wheel_vel = (
            exec_actions[:, self.num_joints : self.num_joints + self.num_wheels] * self.env_cfg["wheel_vel_scale"]
        )

        target_joint_pos = torch.concatenate((target_hip_pos, target_knee_pos), dim=-1) + self.default_joint_pos
        self.target_joint_pos.copy_(target_joint_pos)
        self.target_wheel_vel.copy_(target_wheel_vel)

        assert target_joint_pos.shape[-1] == len(self.joints_dof_idx)
        assert target_wheel_vel.shape[-1] == len(self.wheels_dof_idx)

        self.robot.control_dofs_position(target_joint_pos, self.joints_dof_idx)
        self.robot.control_dofs_velocity(target_wheel_vel, self.wheels_dof_idx)
        self.scene.step()

        ########### 更新buffer ###########
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.init_base_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        # 机器人的速度转化到自身坐标系下
        base_quat_inv = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), base_quat_inv)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), base_quat_inv)
        self.projected_gravity = transform_by_quat(self.global_gravity_dir, base_quat_inv)

        self.joint_pos = self.robot.get_dofs_position(self.joints_dof_idx)
        self.joint_vel = self.robot.get_dofs_velocity(self.joints_dof_idx)
        self.wheel_vel = self.robot.get_dofs_velocity(self.wheels_dof_idx)
        self._update_velocity_estimator()

        knee_pos = self.robot.get_dofs_position(self.knees_dof_idx)
        self.leg_length = self._compute_leg_length(knee_pos)
        self.leg_angle = self._compute_leg_angle(self.joint_pos)
        self._update_tracking_gate()

        ########### 判断终止 ###########
        roll_out = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        pitch_out = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        time_out = self.episode_length_buf > self.max_episode_length
        solver_error = self.scene.rigid_solver.get_error_envs_mask().bool()

        self.terminated_buf = roll_out | pitch_out | solver_error
        self.reset_buf = self.terminated_buf | time_out

        ########### 计算奖励 ###########
        self.reward_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            r = reward_func() * self.reward_scales[name]
            self.reward_buf += r
            self.episode_sums[name] += r
        self.episode_metric_sums["height_gate"] += self.height_gate
        self.episode_metric_sums["attitude_gate"] += self.attitude_gate
        self.episode_metric_sums["tracking_gate"] += self.tracking_gate
        self.episode_metric_sums["tracking_gate_fully_open"] += (self.tracking_gate_raw >= 0.95).to(
            dtype=gs.tc_float
        )
        self.episode_metric_sums["velocity_estimator_abs_error"] += torch.abs(
            self.estimated_base_lin_vel - self.base_lin_vel[:, 0]
        )

        ########### 重采样指令 ###########
        self._resample_commands(self.episode_length_buf % self.resample_step == 0)

        ########### 计算timeout ###########
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        ########### 重置环境（如果需要）###########
        self._reset_idx(self.reset_buf)

        ########### 更新观测值 ###########
        self._update_observations()

        ########### 更新旧值 ###########
        self.last_actions.copy_(self.actions)

        return self.get_observations(), self.reward_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict(
            {
                "policy": self.obs_buf,
                "critic": self.critic_obs_buf,
            },
            batch_size=[self.num_envs],
        )

    def _reset_idx(self, env_idx=None):
        finished_episode_lengths = None
        if env_idx is not None and env_idx.any():
            finished_episode_lengths = self.episode_length_buf[env_idx].clone().clamp_min(1).to(gs.tc_float)

        # 重置机器人状态
        self.robot.set_qpos(self.init_qpos, envs_idx=env_idx, zero_velocity=True, skip_forward=True)

        # 重置buffers
        if env_idx is None:
            # 状态相关
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.projected_gravity.copy_(self.init_projected_gravity_dir)
            self.imu_lin_acc.zero_()
            self.imu_ang_vel.zero_()
            self.forward_kinematic_acc.zero_()
            self.wheel_forward_vel.zero_()
            self.estimated_base_lin_vel.zero_()
            self.height_gate.fill_(1.0)
            self.attitude_gate.fill_(1.0)
            self.tracking_gate_raw.fill_(1.0)
            self.tracking_gate.fill_(1.0)
            self.joint_pos.copy_(self.init_joint_pos)
            self.joint_vel.zero_()
            self.wheel_vel.zero_()
            self.leg_length.copy_(self.init_leg_length)
            self.leg_angle.copy_(self.init_leg_angle)
            # 其他
            self.reset_buf.fill_(True)
            self.actions.zero_()
            self.last_actions.zero_()
            self.target_joint_pos.copy_(self.default_joint_pos)
            self.target_wheel_vel.zero_()
            self.episode_length_buf.zero_()
        else:
            torch.where(env_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(env_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                env_idx[:, None], self.init_projected_gravity_dir, self.projected_gravity, out=self.projected_gravity
            )
            self.height_gate.masked_fill_(env_idx, 1.0)
            self.attitude_gate.masked_fill_(env_idx, 1.0)
            self.tracking_gate_raw.masked_fill_(env_idx, 1.0)
            self.tracking_gate.masked_fill_(env_idx, 1.0)
            torch.where(env_idx[:, None], self.init_joint_pos, self.joint_pos, out=self.joint_pos)
            self.base_lin_vel.masked_fill_(env_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(env_idx[:, None], 0.0)
            self.imu_lin_acc.masked_fill_(env_idx[:, None], 0.0)
            self.imu_ang_vel.masked_fill_(env_idx[:, None], 0.0)
            self.forward_kinematic_acc.masked_fill_(env_idx, 0.0)
            self.wheel_forward_vel.masked_fill_(env_idx, 0.0)
            self.estimated_base_lin_vel.masked_fill_(env_idx, 0.0)
            self.joint_vel.masked_fill_(env_idx[:, None], 0.0)
            self.wheel_vel.masked_fill_(env_idx[:, None], 0.0)
            torch.where(env_idx[:, None], self.init_leg_length, self.leg_length, out=self.leg_length)
            torch.where(env_idx[:, None], self.init_leg_angle, self.leg_angle, out=self.leg_angle)
            self.reset_buf.masked_fill_(env_idx, True)
            self.actions.masked_fill_(env_idx[:, None], 0.0)
            self.last_actions.masked_fill_(env_idx[:, None], 0.0)
            torch.where(env_idx[:, None], self.default_joint_pos, self.target_joint_pos, out=self.target_joint_pos)
            self.target_wheel_vel.masked_fill_(env_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(env_idx, 0)

        # 更新extras和episoded的reward
        if env_idx is not None and env_idx.any():
            self.extras["episode"] = {}
            for key, value in self.episode_sums.items():
                # 形状：[本次结束的环境数量]
                self.extras["episode"]["reward_" + key] = value[env_idx] / self.env_cfg["episode_length_s"]
                # 只清空已经结束的环境
                value.masked_fill_(env_idx, 0.0)
            for key, value in self.episode_metric_sums.items():
                self.extras["episode"]["metric_" + key] = value[env_idx] / finished_episode_lengths
                value.masked_fill_(env_idx, 0.0)
            if self.curriculum.enabled:
                self.extras["episode"]["curriculum_stage"] = torch.full_like(
                    self.reward_buf[env_idx], float(self.curriculum.current_stage_index)
                )
        else:
            # 没有 episode 结束，就不要让 logger 收到虚假的零
            self.extras.pop("episode", None)

            # env_idx=None 表示初始化或手动重置全部环境
            if env_idx is None:
                for value in self.episode_sums.values():
                    value.zero_()
                for value in self.episode_metric_sums.values():
                    value.zero_()

        # 重选指令
        self._resample_commands(env_idx)

    def _update_velocity_estimator(self):
        """更新 actor 使用的前向速度；仿真真值只留给 reward、critic 与诊断。"""
        self.wheel_forward_vel.copy_(
            wheel_forward_velocity(
                self.wheel_vel,
                self.wheel_radius,
                self.wheel_velocity_sign,
            )
        )

        if self.imu is None:
            # 历史配置走该分支，保持原模型行为；这些估计量仅用于统一诊断接口。
            self.imu_lin_acc.zero_()
            self.imu_ang_vel.copy_(self.base_ang_vel)
            self.forward_kinematic_acc.zero_()
            self.estimated_base_lin_vel.copy_(self.base_lin_vel[:, 0])
            return

        imu_data = self.imu.read()
        self.imu_lin_acc.copy_(imu_data.lin_acc)
        self.imu_ang_vel.copy_(imu_data.ang_vel)
        self.forward_kinematic_acc.copy_(
            gravity_compensated_forward_acceleration(
                self.imu_lin_acc,
                self.projected_gravity,
                self.gravity_magnitude,
            )
        )
        self.estimated_base_lin_vel.copy_(
            complementary_forward_velocity_update(
                self.estimated_base_lin_vel,
                self.forward_kinematic_acc,
                self.wheel_forward_vel,
                dt=self.dt,
                wheel_correction_time_constant_s=self.wheel_correction_time_constant_s,
                max_abs_velocity=self.max_estimated_velocity,
            )
        )

    def _update_observations(self):
        # 字典插入顺序就是策略输入的拼接顺序；调整顺序或维度后旧模型将不再兼容。
        if self.policy_velocity_source == "imu_wheel_estimator":
            velocity_components = {
                "estimated_base_lin_vel": self.estimated_base_lin_vel.unsqueeze(-1)
                * self.obs_scales["lin_vel"],  # 1
                "imu_ang_vel": self.imu_ang_vel * self.obs_scales["ang_vel"],  # 3
            }
        else:
            velocity_components = {
                "base_lin_vel": self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                "base_ang_vel": self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            }

        self.obs_components = {
            **velocity_components,
            "projected_gravity": self.projected_gravity,  # 3
            "commands": self.commands * self.commands_scale,  # 3
            "joint_pos_offset": (self.joint_pos - self.default_joint_pos) * self.obs_scales["joint_pos"],
            "joint_vel": self.joint_vel * self.obs_scales["joint_vel"],
            "wheel_vel": self.wheel_vel * self.obs_scales["wheel_vel"],
            "leg_length": self.leg_length * self.obs_scales["leg_length"],  # 2
            "leg_angle": self.leg_angle * self.obs_scales["leg_angle"],  # 2
            "actions": self.actions,
        }
        self.obs_buf = torch.concatenate(tuple(self.obs_components.values()), dim=-1)
        self.critic_obs_components = {
            **self.obs_components,
            "privileged_base_lin_vel": self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
            "privileged_base_ang_vel": self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
        }
        self.critic_obs_buf = torch.concatenate(tuple(self.critic_obs_components.values()), dim=-1)
        return

    def get_observation_components(self, env_idx=0):
        """返回指定环境中组成策略输入的、已经缩放并命名的观测分量。"""
        return {name: value[env_idx].detach() for name, value in self.obs_components.items()}

    def get_velocity_estimator_diagnostics(self, env_idx=0):
        """返回同一环境的真值、轮速估计和融合估计，便于量化部署接口误差。"""
        return {
            "source": self.policy_velocity_source,
            "true_forward_velocity": self.base_lin_vel[env_idx, 0].detach(),
            "estimated_forward_velocity": self.estimated_base_lin_vel[env_idx].detach(),
            "wheel_forward_velocity": self.wheel_forward_vel[env_idx].detach(),
            "imu_specific_force": self.imu_lin_acc[env_idx].detach(),
            "gravity_compensated_forward_acceleration": self.forward_kinematic_acc[env_idx].detach(),
        }

    def get_pd_diagnostics(self, env_idx=0):
        """返回配置的 PD 增益、目标状态，以及由速度计算的阻尼估算项。"""
        joint_vel = self.joint_vel[env_idx].detach()
        wheel_vel = self.wheel_vel[env_idx].detach()
        return {
            "joint_names": tuple(self.env_cfg["joint_names"]),
            "joint_kp": self.joint_kp,
            "joint_kd": self.joint_kd,
            "joint_pos": self.joint_pos[env_idx].detach(),
            "joint_target_pos": self.target_joint_pos[env_idx].detach(),
            "joint_vel": joint_vel,
            # 这里只是 Kd 对应的阻尼分量，不是仿真器测得的完整执行器力矩。
            "joint_kd_damping": -self.joint_kd * joint_vel,
            "wheel_names": tuple(self.env_cfg["wheel_names"]),
            "wheel_kd": self.wheel_kd,
            "wheel_target_vel": self.target_wheel_vel[env_idx].detach(),
            "wheel_vel": wheel_vel,
            "wheel_kd_damping": -self.wheel_kd * wheel_vel,
        }

    def _apply_command_ranges(self, values):
        """应用课程指令范围，并重建采样使用的上下限 Tensor。"""
        allowed = {"lin_vel_range", "ang_vel_range", "min_leg_length_range"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported command range curriculum keys: {sorted(unknown)}")
        for name, limits in values.items():
            if len(limits) != 2 or limits[0] > limits[1]:
                raise ValueError(f"{name} must be [lower, upper], got {limits}")
            self.command_cfg[name] = list(limits)
        self.commands_limit = tuple(
            torch.tensor(items, dtype=gs.tc_float, device=gs.device)
            for items in zip(
                self.command_cfg["lin_vel_range"],
                self.command_cfg["ang_vel_range"],
                self.command_cfg["min_leg_length_range"],
            )
        )

    def _apply_reward_scales(self, values):
        """应用原始奖励权重；除死亡奖励外，运行时权重统一乘以 dt。"""
        for name, raw_scale in values.items():
            reward_function = getattr(self, "_reward_" + name)
            raw_scale = float(raw_scale)
            self.raw_reward_scales[name] = raw_scale
            self.reward_cfg["reward_scales"][name] = raw_scale
            self.reward_scales[name] = raw_scale if name == "death" else raw_scale * self.dt
            self.reward_functions[name] = reward_function
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros(
                    (self.num_envs,), dtype=gs.tc_float, device=gs.device
                )

    def _apply_tracking_gate(self, values):
        """应用速度跟踪门控参数；高度在当前机器人上由双腿平均长度表示。"""
        self.tracking_gate_cfg = merge_tracking_gate_config(self.tracking_gate_cfg, values)
        self.reward_cfg["tracking_gate"] = dict(self.tracking_gate_cfg)

    def _update_tracking_gate(self):
        """根据腿长目标和机身倾角更新平滑 AND 门控。"""
        mean_leg_length = self.leg_length.mean(dim=1)
        leg_length_error = torch.abs(mean_leg_length - self.commands[:, 2])
        self.height_gate = smooth_gate(
            leg_length_error,
            self.tracking_gate_cfg["leg_length_full_error"],
            self.tracking_gate_cfg["leg_length_zero_error"],
        )

        # projected_gravity 的水平分量模长等于 sin(tilt)，可连续表示 roll/pitch 合成倾角。
        sin_tilt = torch.linalg.vector_norm(self.projected_gravity[:, :2], dim=1).clamp(0.0, 1.0)
        tilt = torch.asin(sin_tilt)
        self.attitude_gate = smooth_gate(
            tilt,
            math.radians(self.tracking_gate_cfg["attitude_full_angle_deg"]),
            math.radians(self.tracking_gate_cfg["attitude_zero_angle_deg"]),
        )

        self.tracking_gate_raw = self.height_gate * self.attitude_gate
        gate_floor = self.tracking_gate_cfg["floor"]
        self.tracking_gate = gate_floor + (1.0 - gate_floor) * self.tracking_gate_raw

    def _apply_termination_limits(self, values):
        """应用每一步都会从 env_cfg 读取的姿态终止阈值。"""
        allowed = {"termination_if_roll_greater_than", "termination_if_pitch_greater_than"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported termination curriculum keys: {sorted(unknown)}")
        self.env_cfg.update(values)

    def _apply_action_limits(self, values):
        """应用每一步都会从 env_cfg 读取的动作裁剪范围。"""
        allowed = {"clip_hip_joint_action", "clip_knee_joint_action", "clip_wheel_action"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported action-limit curriculum keys: {sorted(unknown)}")
        self.env_cfg.update(values)

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limit, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)
        return

    def _compute_leg_length(self, knee_pos):
        upper_leg_length = 0.17
        lower_leg_length = 0.17

        leg_length_squared = (
            upper_leg_length**2 + lower_leg_length**2 + 2 * upper_leg_length * lower_leg_length * torch.cos(knee_pos)
        )

        return torch.sqrt(torch.clamp_min(leg_length_squared, 0.0))

    def _compute_leg_angle(self, joint_pos):
        hip_pos = joint_pos[..., : self.num_hip_joints]  # (num_envs, 2)
        knee_pos = joint_pos[..., self.num_hip_joints : self.num_hip_joints + self.num_knee_joints]  # (num_envs, 2)

        upper_leg_length = 0.17
        lower_leg_length = 0.17

        leg_x = -upper_leg_length * torch.sin(hip_pos) - lower_leg_length * torch.sin(hip_pos + knee_pos)

        leg_down = upper_leg_length * torch.cos(hip_pos) + lower_leg_length * torch.cos(hip_pos + knee_pos)

        leg_angle = torch.atan2(leg_x, leg_down)
        return leg_angle

    # ----------奖励函数------------
    # 这里的奖励只计算相对大小，缩放和正负由reward_scales决定

    def _reward_tracking_lin_vel(self):
        # 弱的无门控误差惩罚，确保门控未打开时仍有速度学习信号。
        lin_vel_error = torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])
        return lin_vel_error

    def _reward_tracking_ang_vel(self):
        # 弱的无门控误差惩罚，确保门控未打开时仍有角速度学习信号。
        ang_vel_error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])
        return ang_vel_error

    def _reward_gated_tracking_lin_vel(self):
        lin_vel_error = torch.square(self.base_lin_vel[:, 0] - self.commands[:, 0])
        tracking_bonus = torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
        return self.tracking_gate * tracking_bonus

    def _reward_gated_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 1])
        tracking_bonus = torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])
        return self.tracking_gate * tracking_bonus

    def _reward_base_balance(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_roll_balance(self):
    #     return torch.square(self.base_euler[:, 0])

    def _reward_joint_vel(self):
        # 惩罚关节速度
        joint_vel_error = torch.sum(torch.square(self.joint_vel), dim=1)
        return joint_vel_error

    def _reward_joint_pos(self):
        # 惩罚关节位置
        joint_pos_error = torch.sum(
            torch.square(self.joint_pos - self.default_joint_pos),
            dim=-1,
        )
        return joint_pos_error

    def _reward_leg_symmetry(self):
        return torch.square(self.leg_angle[:, 0] - self.leg_angle[:, 1])

    def _reward_leg_length(self):
        mean_leg_length = self.leg_length.mean(dim=1) * self.obs_scales["leg_length"]
        error = mean_leg_length - self.commands[:, 2] * self.obs_scales["leg_length"]
        return torch.square(error)

    def _reward_alive(self):
        return torch.ones(
            self.num_envs,
            dtype=gs.tc_float,
            device=self.device,
        )

    def _reward_death(self):
        return self.terminated_buf.to(dtype=gs.tc_float)


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def _gain_tensor(value, count, name):
    if isinstance(value, (int, float)):
        values = [float(value)] * count
    else:
        values = list(value)
        if len(values) != count:
            raise ValueError(f"{name} requires {count} values, got {len(values)}")
    return torch.tensor(values, dtype=gs.tc_float, device=gs.device)
