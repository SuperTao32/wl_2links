"""训练版本的创建、参数保存，以及 eval 自动发现工具。"""

from __future__ import annotations

import json
import pickle
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping


RUN_DIR_PATTERN = re.compile(r"^version_(\d+)$")
CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")
CONFIG_FILENAME = "cfgs.pkl"


def create_versioned_run_dir(log_root: str | Path, exp_name: str) -> Path:
    """为一次训练原子地创建下一个 ``version_NNNN`` 目录。"""
    experiment_dir = Path(log_root) / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    versions = [_version_number(path) for path in experiment_dir.iterdir() if path.is_dir()]
    next_version = max((number for number in versions if number is not None), default=-1) + 1

    # exist_ok=False 保证并发启动训练时不会共用目录：抢到同一版本号失败的
    # 进程会继续尝试下一个编号，也不会删除另一进程已经创建的日志。
    while True:
        run_dir = experiment_dir / f"version_{next_version:04d}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            next_version += 1


def resolve_run_dir(
    log_root: str | Path,
    exp_name: str,
    version: str | int | None = None,
    *,
    require_checkpoint: bool = False,
) -> Path:
    """解析指定版本；未指定时返回编号最大的可用训练版本。

    保留对旧式 ``logs/<experiment>`` 非版本化目录的兼容。自动选择时，
    ``require_checkpoint=True`` 会跳过只有配置、没有模型的中断训练。
    """
    experiment_dir = Path(log_root) / exp_name
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment log directory does not exist: {experiment_dir}")

    if version is not None:
        version_name = str(version)
        if version_name.isdigit():
            version_name = f"version_{int(version_name):04d}"
        run_dir = experiment_dir / version_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training version does not exist: {run_dir}")
        _validate_run_dir(run_dir, require_checkpoint=require_checkpoint)
        return run_dir

    # 只识别 version_数字，其他人工创建的目录不会影响“最新版本”的判断。
    candidates = sorted(
        (
            (number, path)
            for path in experiment_dir.iterdir()
            if path.is_dir() and (number := _version_number(path)) is not None
        ),
        reverse=True,
    )
    for _, run_dir in candidates:
        if _is_usable_run(run_dir, require_checkpoint=require_checkpoint):
            return run_dir

    # 新版本目录都不可用时，再尝试原来的 logs/<experiment>/cfgs.pkl 布局。
    if _is_usable_run(experiment_dir, require_checkpoint=require_checkpoint):
        return experiment_dir

    requirement = "configuration and checkpoint" if require_checkpoint else "configuration"
    raise FileNotFoundError(f"No run with a valid {requirement} found in: {experiment_dir}")


def resolve_checkpoint(run_dir: str | Path, checkpoint: int | None = None) -> Path:
    """解析指定 checkpoint；未指定时返回编号最大的 ``model_N.pt``。"""
    run_dir = Path(run_dir)
    if checkpoint is not None:
        checkpoint_path = run_dir / f"model_{checkpoint}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoints = sorted(
        (
            (number, path)
            for path in run_dir.iterdir()
            if path.is_file() and (number := _checkpoint_number(path)) is not None
        ),
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")
    return checkpoints[0][1]


def save_run_artifacts(
    run_dir: str | Path,
    configs: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    """同时保存可精确恢复的 pickle、便于查看的 JSON 和运行元数据。"""
    run_dir = Path(run_dir)
    # schema_version 用于以后升级配置结构时继续兼容历史训练结果。
    payload = {"schema_version": 2, **dict(configs)}

    with (run_dir / CONFIG_FILENAME).open("wb") as file:
        pickle.dump(payload, file)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)

    metadata = _collect_metadata(run_dir, arguments)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=_json_default)


def load_run_configs(run_dir: str | Path) -> dict[str, Any]:
    """加载当前字典格式，同时兼容历史的五项或六项列表格式。"""
    config_path = Path(run_dir) / CONFIG_FILENAME
    with config_path.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict):
        required = {"env_cfg", "obs_cfg", "reward_cfg", "command_cfg", "train_cfg"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Config is missing required sections {sorted(missing)}: {config_path}")
        payload.setdefault("curriculum_cfg", {"enabled": False, "stages": []})
        return payload

    # 最早的日志没有 curriculum_cfg，读取时补成默认关闭状态。
    if isinstance(payload, (list, tuple)) and len(payload) == 5:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = payload
        return {
            "schema_version": 1,
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "curriculum_cfg": {"enabled": False, "stages": []},
            "train_cfg": train_cfg,
        }

    if isinstance(payload, (list, tuple)) and len(payload) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, train_cfg = payload
        return {
            "schema_version": 2,
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "curriculum_cfg": curriculum_cfg,
            "train_cfg": train_cfg,
        }

    raise ValueError(f"Unsupported config format in: {config_path}")


def _collect_metadata(run_dir: Path, arguments: Mapping[str, Any]) -> dict[str, Any]:
    # 配置决定实验参数；提交号、dirty 状态和依赖版本用于定位代码运行环境。
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "arguments": dict(arguments),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions("genesis-world", "rsl-rl-lib", "torch"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_versions(*names: str) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_run_dir(run_dir: Path, *, require_checkpoint: bool) -> None:
    if not (run_dir / CONFIG_FILENAME).is_file():
        raise FileNotFoundError(f"Run configuration does not exist: {run_dir / CONFIG_FILENAME}")
    if require_checkpoint and not any(
        path.is_file() and _checkpoint_number(path) is not None for path in run_dir.iterdir()
    ):
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")


def _is_usable_run(run_dir: Path, *, require_checkpoint: bool) -> bool:
    try:
        _validate_run_dir(run_dir, require_checkpoint=require_checkpoint)
    except FileNotFoundError:
        return False
    return True


def _version_number(path: Path) -> int | None:
    match = RUN_DIR_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _checkpoint_number(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return repr(value)
