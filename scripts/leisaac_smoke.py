"""Launch one SO-101 LeIsaac environment for a short headless smoke check."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("LeIsaac-SO101-LiftCube-v0", "LeIsaac-SO101-PickOrange-v0"),
                    required=True)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import leisaac.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

try:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset(seed=20261010)
        print(f"RESET_OK task={args.task} obs_type={type(obs).__name__}", flush=True)
        for index in range(args.steps):
            action = torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)
        print(f"STEP_OK task={args.task} steps={args.steps} reward={reward}", flush=True)
    finally:
        env.close()
finally:
    simulation_app.close()
