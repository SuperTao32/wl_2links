import argparse
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
import numpy as np
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from run_utils import create_versioned_run_dir, save_run_artifacts
from wl_env import Wl_Env


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            # "distribution_cfg": {
            #     "class_name": "GaussianDistribution",
            #     "init_std": 1.0,
            #     "std_type": "scalar",
            # },
            "distribution_cfg": {
                "class_name": "BetaDistribution",
                "action_range": (-1.0, 1.0),
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            # critic 只在训练阶段存在，可以读取仿真真值；部署导出的 actor 仍只依赖 policy。
            "critic": ["critic"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        "num_joints": 4,
        "num_hip_joints": 2,
        "num_knee_joints": 2,
        "num_wheels": 2,
        # joint/link names
        "default_joint_pos": {  # [rad]
            "left_hip": np.pi / 4,
            "right_hip": np.pi / 4,
            "left_knee": -np.pi / 2,
            "right_knee": -np.pi / 2,
        },
        "joint_names": [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        ],
        "wheel_names": [
            "left_wheel_joint",
            "right_wheel_joint",
        ],
        # PD
        "joint_kp": 20.0,
        "joint_kd": 0.6,
        "wheel_kd": 0.3,
        # termination
        "termination_if_roll_greater_than": 15,  # degree
        "termination_if_pitch_greater_than": 15,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.325],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 100.0,
        "resampling_time_s": 10.0,
        "hip_joint_pos_scale": np.pi / 4,
        "knee_joint_pos_scale": np.pi / 3,
        "wheel_vel_scale": 20.0,
        "simulate_action_latency": True,
        "clip_hip_joint_action": 1.0,
        "clip_knee_joint_action": 1.0,
        "clip_wheel_action": 1.0,
    }
    obs_cfg = {
        # 新训练使用真机可复现的速度接口。历史 cfg 没有该字段时，环境会退回原来的
        # simulator 模式，从而保持旧 checkpoint 的 32 维输入和观测顺序。
        "policy_velocity_source": "imu_wheel_estimator",
        "imu": {
            "link_name": "base_link",
            "pos_offset": [0.0, 0.0, 0.0],
            "acc_noise": 0.005,
            # 以下误差项先保留配置入口；正式 sim2real 前应替换成实测标定结果。
            "acc_bias": 0.0,
            "acc_random_walk": 0.0,
            "gyro_noise": 0.0,
            "gyro_bias": 0.0,
            "gyro_random_walk": 0.0,
            "delay": 0.0,
            "jitter": 0.0,
        },
        "velocity_estimator": {
            "wheel_radius": 0.08,
            # 当前 URDF/Genesis 中正轮速对应机体 -x；真机应按编码器与
            # 车体坐标定义重新确认。
            "wheel_velocity_sign": 1.0,
            "gravity_magnitude": 9.81,
            # 0.5 s 是便于起步验证的基线：IMU 决定短时变化，轮速逐渐消除积分漂移。
            # 真机发生明显打滑时应依据日志重新标定，不能把这个值视为最终参数。
            "wheel_correction_time_constant_s": 0.5,
            "max_abs_velocity": 3.0,
        },
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "joint_pos": 1.0,
            "joint_vel": 1.0,
            "wheel_vel": 1.0 / 20.0,
            "leg_length": 1.0 / 0.34,
            "leg_angle": 1.0,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        # 高度门控当前使用双腿平均长度与 commands[:, 2] 的误差。
        "tracking_gate": {
            "leg_length_full_error": 0.02,
            "leg_length_zero_error": 0.05,
            "attitude_full_angle_deg": 5.0,
            "attitude_zero_angle_deg": 11.0,
            "floor": 0.10,
        },
        "reward_scales": {
            # 无门控平方误差只保留较弱权重；主要收益来自下方的门控正奖励。
            "tracking_lin_vel": -2.0,
            "tracking_ang_vel": -2.0,
            "gated_tracking_lin_vel": 20.0,
            "gated_tracking_ang_vel": 20.0,
            "base_balance": -30.0,
            # "roll_balance": -5.0,
            "leg_symmetry": -10.0,
            "leg_length": -20.0,
            "joint_vel": -1.0,
            "joint_pos": 0,
            "alive": 5.0,
            "death": -100.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_range": [-1.0, 1.0],
        "ang_vel_range": [-0.8, 0.8],
        "min_leg_length_range": [0.15, 0.3],
    }
    curriculum_cfg = {
        "enabled": True,
        "update_every_steps": 24,
        "stages": [
            {
                "name": "balance0",
                "start_step": 0,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-0.10, 0.10],
                        "ang_vel_range": [-0.2, 0.2],
                        "min_leg_length_range": [0.24, 0.24],
                    },
                    "tracking_gate": {
                        "leg_length_full_error": 0.025,
                        "leg_length_zero_error": 0.06,
                        "attitude_full_angle_deg": 5.0,
                        "attitude_zero_angle_deg": 10.0,
                        "floor": 0.30,
                    },
                    "termination_limits": {
                        "termination_if_roll_greater_than": 15,
                        "termination_if_pitch_greater_than": 15,
                    },
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -2.0,
                        "gated_tracking_lin_vel": 15.0,
                        "gated_tracking_ang_vel": 15.0,
                        "base_balance": -30.0,
                        "leg_symmetry": -20.0,
                        "leg_length": -30.0,
                        "joint_vel": -1.0,
                        "joint_pos": 0,
                        "alive": 20.0,
                        "death": -100.0,
                    },
                },
            },
            {
                "name": "balance1",
                "start_step": 9600,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-0.5, 0.5],
                        "ang_vel_range": [-0.4, 0.4],
                        "min_leg_length_range": [0.22, 0.26],
                    },
                    "tracking_gate": {
                        "leg_length_full_error": 0.02,
                        "leg_length_zero_error": 0.05,
                        "attitude_full_angle_deg": 4.0,
                        "attitude_zero_angle_deg": 8.0,
                        "floor": 0.20,
                    },
                    "termination_limits": {
                        "termination_if_roll_greater_than": 15,
                        "termination_if_pitch_greater_than": 15,
                    },
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -2.0,
                        "gated_tracking_lin_vel": 20.0,
                        "gated_tracking_ang_vel": 20.0,
                        "base_balance": -35.0,
                        "leg_symmetry": -10.0,
                        "leg_length": -40.0,
                        "joint_vel": -1.0,
                        "joint_pos": 0,
                        "alive": 15.0,
                        "death": -100.0,
                    },
                },
            },
            {
                "name": "balance2",
                "start_step": 16800,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-0.75, 0.75],
                        "ang_vel_range": [-0.6, 0.6],
                        "min_leg_length_range": [0.18, 0.28],
                    },
                    "tracking_gate": {
                        "leg_length_full_error": 0.015,
                        "leg_length_zero_error": 0.04,
                        "attitude_full_angle_deg": 2.0,
                        "attitude_zero_angle_deg": 6.0,
                        "floor": 0.10,
                    },
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -2.0,
                        "gated_tracking_lin_vel": 30.0,
                        "gated_tracking_ang_vel": 30.0,
                        "base_balance": -40.0,
                        "leg_symmetry": -10.0,
                        "leg_length": -50.0,
                        "joint_vel": -1.0,
                        "joint_pos": 0,
                        "alive": 10.0,
                        "death": -100.0,
                    },
                },
            },
            {
                "name": "full_range",
                "start_step": 28800,
                "targets": {
                    "command_ranges": {
                        "lin_vel_range": [-1.0, 1.0],
                        "ang_vel_range": [-0.8, 0.8],
                        "min_leg_length_range": [0.16, 0.28],
                    },
                    "tracking_gate": {
                        "leg_length_full_error": 0.01,
                        "leg_length_zero_error": 0.025,
                        "attitude_full_angle_deg": 1.0,
                        "attitude_zero_angle_deg": 3.0,
                        "floor": 0.05,
                    },
                    "reward_scales": {
                        "tracking_lin_vel": -2.0,
                        "tracking_ang_vel": -2.0,
                        "gated_tracking_lin_vel": 30.0,
                        "gated_tracking_ang_vel": 30.0,
                        "base_balance": -40.0,
                        "leg_symmetry": -10.0,
                        "leg_length": -50.0,
                        "joint_vel": -1.0,
                        "joint_pos": 0,
                        "alive": 5.0,
                        "death": -100.0,
                    },
                },
            },
        ],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-e", "--exp_name", type=str, default="wl_2link_1.4")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=2001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-root", type=str, default="logs")
    args = parser.parse_args()

    # 每次启动都新建版本，旧日志和 checkpoint 不会被覆盖。
    run_dir = create_versioned_run_dir(args.log_root, args.exp_name)
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)
    train_cfg["run_name"] = f"{args.exp_name}/{run_dir.name}"

    configs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "curriculum_cfg": curriculum_cfg,
        "train_cfg": train_cfg,
    }
    # 在初始化 Genesis 前先落盘配置；即使后续初始化失败，也能保留失败现场。
    save_run_artifacts(run_dir, configs, vars(args))
    print(f"[train] saving this run to: {run_dir}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Wl_Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
