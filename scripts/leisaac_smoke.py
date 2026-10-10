"""Launch one SO-101 LeIsaac environment for a short headless smoke check."""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("LeIsaac-SO101-LiftCube-v0", "LeIsaac-SO101-LiftCube-Direct-v0", "LeIsaac-SO101-PickOrange-v0"),
                    required=True)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app
print("APP_LAUNCHED", flush=True)

import gymnasium as gym  # noqa: E402
print("GYM_IMPORTED", flush=True)
import leisaac.tasks  # noqa: E402,F401
print("LEISAAC_TASKS_IMPORTED", flush=True)
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
print("TASK_UTILS_IMPORTED", flush=True)

try:
    print(f"PARSING_TASK task={args.task}", flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    print("TASK_PARSED", flush=True)
    env = gym.make(args.task, cfg=cfg)
    print("ENV_CREATED", flush=True)
    try:
        obs, _ = env.reset(seed=20261010)
        print(f"RESET_OK task={args.task} obs_type={type(obs).__name__}", flush=True)
        for index in range(args.steps):
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)
        print(f"STEP_OK task={args.task} steps={args.steps} reward={reward}", flush=True)
    finally:
        env.close()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
