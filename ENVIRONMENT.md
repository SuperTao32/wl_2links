# 环境依赖管理指南

本项目是一个使用 Genesis、MuJoCo 和 RSL-RL 的实验项目，不作为 Python
包发布。环境由 `uv` 管理：

- `pyproject.toml` 声明项目直接依赖及允许的版本范围。
- `.python-version` 指定 Python 3.13。
- `uv.lock` 保存完整、精确的依赖解析结果。
- `.venv/` 是本机虚拟环境，不提交到 Git。

`[tool.uv] package = false` 表示 uv 只管理依赖和虚拟环境，不构建或安装
`wheel-leg` 本身。

## 首次获取项目

### 1. 安装 uv

先按照 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)
安装 uv，然后确认命令可用：

```bash
uv --version
```

在 macOS 上也可以使用 Homebrew：

```bash
brew install uv
```

### 2. 克隆并安装环境

```bash
git clone <repository-url>
cd wl_test
uv sync --locked
```

`uv sync --locked` 会：

1. 根据 `.python-version` 选择 Python 3.13；
2. 创建项目本地的 `.venv/`；
3. 严格按照已提交的 `uv.lock` 安装依赖；
4. 在 `pyproject.toml` 与锁文件不一致时停止并报错。

不需要手动执行 `python -m venv`、`source .venv/bin/activate` 或
`pip install`。

## 获取项目更新后同步环境

`git fetch` 只下载远端提交，不会修改当前工作区。先将远端更新整合到当前
分支，再同步环境：

```bash
git fetch origin
git pull --ff-only
uv sync --locked
```

如果切换到了其他已存在的分支：

```bash
git switch <branch-name>
uv sync --locked
```

只要新的提交修改了 `pyproject.toml` 或 `uv.lock`，就应重新执行
`uv sync --locked`。

## 运行实验

从仓库根目录运行命令，确保 `assets/...` 和 `logs/...` 等相对路径正确：

```bash
uv run --locked python experiments/genesis/wheel_leg/wl_train.py --help
uv run --locked python experiments/genesis/go2/go2_train.py --help
uv run --locked python experiments/mujoco/view.py
```

`uv run --locked` 会使用项目的 `.venv`，因此通常不需要先激活虚拟环境。

## 新增依赖

不要直接在 `.venv` 中执行 `pip install`。使用 `uv add`，让依赖声明和锁
文件一起更新：

```bash
uv add matplotlib
```

如果已知兼容范围，可以明确声明：

```bash
uv add "matplotlib>=3.10,<4"
```

`uv add` 会更新：

- `pyproject.toml`；
- `uv.lock`；
- 当前 `.venv`。

只应把代码直接使用的库声明为直接依赖。间接依赖由 uv 解析并记录在
`uv.lock` 中。

## 删除依赖

```bash
uv remove matplotlib
```

不要只从 `pyproject.toml` 手工删除后就结束；应确保 `uv.lock` 和
`.venv` 也同步更新。

## 升级依赖

只升级一个包：

```bash
uv lock --upgrade-package mujoco
uv sync
```

升级全部允许范围内的依赖：

```bash
uv lock --upgrade
uv sync
```

升级后必须运行最小验证，确认 Genesis、MuJoCo、Torch 和 RSL-RL 仍然
兼容，再提交新的锁文件。不要为了追求“最新”而无验证地升级全部依赖。

## 手工修改 pyproject.toml 后

如果手工调整了依赖或版本范围，执行：

```bash
uv lock
uv sync
uv lock --check
```

随后检查解析结果：

```bash
uv tree --depth 1
```

## 环境验证

验证核心包是否从项目虚拟环境导入：

```bash
uv run --locked python -c "
import sys
import genesis
import mujoco
import torch
import rsl_rl
print('Python:', sys.executable)
print('Genesis:', genesis.__file__)
print('Genesis has init:', hasattr(genesis, 'init'))
print('MuJoCo:', mujoco.__file__)
print('Torch:', torch.__version__)
print('RSL-RL:', rsl_rl.__file__)
"
```

输出路径应位于本项目的 `.venv/` 中，且 `Genesis has init` 应为 `True`。

再验证实验脚本的导入链：

```bash
uv run --locked python experiments/genesis/wheel_leg/wl_train.py --help
uv run --locked python experiments/genesis/go2/go2_train.py --help
```

`--help` 会在真正创建仿真环境或开始训练前退出，适合作为快速检查。

## 提交依赖变更

依赖发生变化时，通常应同时提交：

```text
pyproject.toml
uv.lock
```

Python 版本变化时还要提交：

```text
.python-version
```

提交前检查：

```bash
uv lock --check
git diff -- pyproject.toml uv.lock .python-version
git status --short
```

不要提交：

```text
.venv/
__pycache__/
*.egg-info/
logs/
```

## 文件职责速查

| 文件 | 职责 | 是否提交 |
|---|---|---|
| `pyproject.toml` | 声明直接依赖和兼容范围 | 是 |
| `uv.lock` | 锁定全部直接与间接依赖 | 是 |
| `.python-version` | 固定 Python 3.13 | 是 |
| `.venv/` | 当前机器的实际虚拟环境 | 否 |

## 维护原则

1. 使用 `uv add` 和 `uv remove` 管理依赖。
2. 不手工编辑 `uv.lock`。
3. 不使用 `pip freeze` 覆盖项目依赖声明。
4. 不把本机 `.venv` 复制给其他人。
5. 更新依赖后先验证，再提交 `pyproject.toml` 和 `uv.lock`。
6. 其他人获取含依赖变更的提交后，执行 `uv sync --locked`。
