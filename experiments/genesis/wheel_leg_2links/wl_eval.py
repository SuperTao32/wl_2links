import argparse
from copy import deepcopy
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.ext.pyrender.constants import FONT_SIZE, TEXT_PADDING, TextAlign
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins import ViewerPlugin

from run_utils import load_run_configs, resolve_checkpoint, resolve_run_dir
from wl_env import Wl_Env


class DiagnosticsOverlay(ViewerPlugin):
    """Draw persistent evaluation diagnostics as separate lines."""

    font_size = 25

    def __init__(self):
        super().__init__()
        self._lines = ()

    def update(self, lines):
        # The simulation and viewer run in different threads. Replacing an
        # immutable tuple keeps the viewer from observing a partially updated list.
        self._lines = tuple(lines)

    def on_draw(self):
        viewer = self.viewer
        lines = self._lines
        if viewer is None or viewer._renderer is None or not lines:
            return

        x = viewer._viewport_size[0] - TEXT_PADDING
        # Leave the first row free for Genesis' keyboard-help prompt.
        y = viewer._viewport_size[1] - TEXT_PADDING - int(FONT_SIZE * 1.2)
        line_height = int(self.font_size * 1.2)
        for index, line in enumerate(lines):
            viewer._renderer.render_text(
                line,
                x,
                y - index * line_height,
                font_name="SpaceMono-Regular",
                font_pt=self.font_size,
                color=viewer._font_color,
                align=TextAlign.TOP_RIGHT,
            )


class KeyboardCommand:
    """保存键盘指令；满足一个 Viewer 线程和一个仿真线程的使用场景。"""

    def __init__(self, env):
        self.env = env
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.leg_length = float(env.init_leg_length.mean().item())
        self.lin_step = 0.1
        self.ang_step = 0.1
        self.height_step = 0.01

        viewer = env.scene.viewer
        self.viewer = viewer
        self.diagnostics_overlay = viewer.add_plugin(DiagnosticsOverlay())
        viewer.register_keybinds(
            self._keybind("command_forward", Key.UP, self._change_lin, self.lin_step),
            self._keybind("command_backward", Key.DOWN, self._change_lin, -self.lin_step),
            self._keybind("command_turn_left", Key.LEFT, self._change_ang, self.ang_step),
            self._keybind("command_turn_right", Key.RIGHT, self._change_ang, -self.ang_step),
            self._keybind("command_raise_body", Key.PAGEUP, self._change_height, self.height_step),
            self._keybind("command_lower_body", Key.PAGEDOWN, self._change_height, -self.height_step),
            Keybind(
                "command_stop",
                Key.SPACE,
                key_action=KeyAction.PRESS,
                callback=self.stop,
            ),
        )

    @staticmethod
    def _keybind(name, key, callback, amount):
        return Keybind(
            name,
            key,
            key_action=KeyAction.PRESS,
            callback=callback,
            args=(amount,),
        )

    def _change_lin(self, amount):
        lower, upper = self.env.command_cfg["lin_vel_range"]
        self.lin_vel = min(max(self.lin_vel + amount, lower), upper)

    def _change_ang(self, amount):
        lower, upper = self.env.command_cfg["ang_vel_range"]
        self.ang_vel = min(max(self.ang_vel + amount, lower), upper)

    def _change_height(self, amount):
        lower, upper = self.env.command_cfg["min_leg_length_range"]
        self.leg_length = min(max(self.leg_length + amount, lower), upper)

    def stop(self):
        self.lin_vel = 0.0
        self.ang_vel = 0.0

    def write_to_env(self):
        self.env.commands[:, 0] = self.lin_vel
        self.env.commands[:, 1] = self.ang_vel
        self.env.commands[:, 2] = self.leg_length

    def update_caption(self):
        env = self.env
        obs = env.obs_buf[0]
        pd = env.get_pd_diagnostics()
        velocity = env.get_velocity_estimator_diagnostics()
        text = (
            f"command  vx={self.lin_vel:+.2f} m/s  wz={self.ang_vel:+.2f} rad/s  leg={self.leg_length:.3f} m",
            f"state    vx_true={env.base_lin_vel[0, 0].item():+.2f} m/s  "
            f"vx_est={velocity['estimated_forward_velocity'].item():+.2f} m/s  "
            f"wz={env.base_ang_vel[0, 2].item():+.2f} rad/s",
            f"pose     roll={env.base_euler[0, 0].item():+.1f} deg  pitch={env.base_euler[0, 1].item():+.1f} deg",
            f"legs     left={env.leg_length[0, 0].item():.3f} m  right={env.leg_length[0, 1].item():.3f} m",
            f"obs      n={obs.numel()}  min={obs.min().item():+.2f}  max={obs.max().item():+.2f}  "
            f"mean={obs.mean().item():+.2f}",
            f"joint Kd       {_format_tensor(pd['joint_kd'], precision=2)}",
            f"joint -Kd*qdot {_format_tensor(pd['joint_kd_damping'], precision=2)}",
        )
        self.diagnostics_overlay.update(text)


def _format_tensor(tensor, precision=3):
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:+.{precision}f}" for value in values) + "]"


def print_evaluation_diagnostics(env, step):
    """按观测分组打印策略输入，并输出关节与轮子的 PD/Kd 诊断。"""
    print(
        f"\n[eval step {step}] scaled observations "
        f"(policy={env.obs_buf.shape[-1]}, critic={env.critic_obs_buf.shape[-1]})"
    )
    for name, value in env.get_observation_components().items():
        print(f"  obs.{name:<20} {_format_tensor(value)}")

    velocity = env.get_velocity_estimator_diagnostics()
    print(f"  velocity source          {velocity['source']}")
    print(f"  velocity true            {velocity['true_forward_velocity'].item():+.4f} m/s")
    print(f"  velocity estimated       {velocity['estimated_forward_velocity'].item():+.4f} m/s")
    print(f"  velocity from wheels     {velocity['wheel_forward_velocity'].item():+.4f} m/s")
    print(f"  IMU specific force       {_format_tensor(velocity['imu_specific_force'])} m/s^2")
    print(
        "  acceleration corrected  "
        f"{velocity['gravity_compensated_forward_acceleration'].item():+.4f} m/s^2"
    )

    pd = env.get_pd_diagnostics()
    print("  joint names             " + ", ".join(pd["joint_names"]))
    print(f"  joint Kp                {_format_tensor(pd['joint_kp'])}")
    print(f"  joint Kd                {_format_tensor(pd['joint_kd'])}")
    print(f"  joint target position   {_format_tensor(pd['joint_target_pos'])}")
    print(f"  joint position          {_format_tensor(pd['joint_pos'])}")
    print(f"  joint velocity qdot     {_format_tensor(pd['joint_vel'])}")
    print(f"  joint -Kd*qdot (estimate) {_format_tensor(pd['joint_kd_damping'])}")
    print("  wheel names             " + ", ".join(pd["wheel_names"]))
    print(f"  wheel Kd                {_format_tensor(pd['wheel_kd'])}")
    print(f"  wheel target velocity   {_format_tensor(pd['wheel_target_vel'])}")
    print(f"  wheel velocity qdot     {_format_tensor(pd['wheel_vel'])}")
    print(f"  wheel -Kd*qdot (estimate) {_format_tensor(pd['wheel_kd_damping'])}", flush=True)


def verify_fixed_command(
    env,
    policy,
    command_x,
    total_steps=300,
    warmup_steps=50,
):
    fixed_commands = torch.tensor(
        [[command_x, 0.0, 0.25]],
        dtype=gs.tc_float,
        device=gs.device,
    ).expand(env.num_envs, -1)

    vx_samples = []
    wz_samples = []

    env.reset()

    with torch.no_grad():
        for step in range(total_steps):
            # 环境内部可能重新采样指令，因此每一步都覆盖
            env.commands.copy_(fixed_commands)

            # commands 是观测的一部分，修改后必须重新构造观测
            env._update_observations()
            obs_dict = env.get_observations()

            actions = policy(obs_dict)
            _, _, dones, _ = env.step(actions)

            if step >= warmup_steps:
                valid = ~dones

                if valid.any():
                    vx_samples.append(env.base_lin_vel[valid, 0].detach().cpu())
                    wz_samples.append(env.base_ang_vel[valid, 2].detach().cpu())

    vx = torch.cat(vx_samples)
    wz = torch.cat(wz_samples)

    return {
        "command_x": command_x,
        "vx_mean": vx.mean().item(),
        "vx_std": vx.std().item(),
        "tracking_rmse": torch.sqrt(torch.mean((vx - command_x) ** 2)).item(),
        "wz_mean": wz.mean().item(),
        "wz_rms": torch.sqrt(torch.mean(wz**2)).item(),
        "num_samples": vx.numel(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="wl_2link_1.4")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--version", type=str, default=None, help="version_0003 or 3; default: latest valid run")
    parser.add_argument("--ckpt", type=int, default=None, help="checkpoint number; default: latest in selected run")
    parser.add_argument(
        "--print-interval",
        type=int,
        default=50,
        help="print named observations and PD/Kd diagnostics every N steps; 0 disables periodic output",
    )
    args = parser.parse_args()

    if args.print_interval < 0:
        parser.error("--print-interval cannot be negative")

    # 默认选择“含配置且含 checkpoint”的最大版本号，自动跳过中断训练。
    run_dir = resolve_run_dir(
        args.log_root,
        args.exp_name,
        args.version,
        require_checkpoint=True,
    )
    checkpoint_path = resolve_checkpoint(run_dir, args.ckpt)
    configs = load_run_configs(run_dir)
    env_cfg = configs["env_cfg"]
    obs_cfg = configs["obs_cfg"]
    reward_cfg = deepcopy(configs["reward_cfg"])
    command_cfg = configs["command_cfg"]
    train_cfg = configs["train_cfg"]
    reward_cfg["reward_scales"] = {}

    print(f"[eval] run:        {run_dir}")
    print(f"[eval] checkpoint: {checkpoint_path}")

    gs.init(backend=gs.gpu)

    env = Wl_Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        # Eval 直接使用 command_cfg 中保存的完整任务范围，不从课程初级阶段重跑。
        curriculum_cfg={"enabled": False, "stages": []},
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, str(run_dir), device=gs.device)
    runner.load(str(checkpoint_path), map_location=gs.device)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    keyboard = KeyboardCommand(env)
    print_evaluation_diagnostics(env, step=0)
    step = 0
    with torch.no_grad():
        while env.scene.viewer.is_alive():
            # step() 内部可能重采样指令，因此每次策略推理前都覆盖为键盘指令。
            keyboard.write_to_env()
            env._update_observations()
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            keyboard.update_caption()
            step += 1
            if args.print_interval and step % args.print_interval == 0:
                print_evaluation_diagnostics(env, step)

    # checkpoints = [100, 300, 500, 1000]
    # commands = [-1.0, 0.0, 1.0]

    # for ckpt in checkpoints:
    #     runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    #     policy = runner.get_inference_policy(device=gs.device)

    #     print(f"\ncheckpoint: {ckpt}")

    #     for command_x in commands:
    #         result = verify_fixed_command(
    #             env=env,
    #             policy=policy,
    #             command_x=command_x,
    #             total_steps=300,
    #             warmup_steps=50,
    #         )

    #         print(
    #             f"command={result['command_x']:+.1f} | "
    #             f"vx={result['vx_mean']:+.4f} ± "
    #             f"{result['vx_std']:.4f} | "
    #             f"tracking_rmse={result['tracking_rmse']:.4f} | "
    #             f"wz_mean={result['wz_mean']:+.4f} | "
    #             f"wz_rms={result['wz_rms']:.4f}"
    #         )


if __name__ == "__main__":
    main()
