"""Discrete keyboard teleoperation for collecting SO-101 demonstrations."""

import argparse
import queue
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from .demonstrations import save_demonstrations
from .env import OBJECTS, SO101PickPlaceEnv


KEY_ACTIONS = {
    "W": (0, 1), "S": (0, -1), "D": (1, 1), "A": (1, -1),
    "R": (2, 1), "F": (2, -1), "J": (3, 1), "L": (3, -1),
    "I": (4, 1), "U": (4, -1), "O": (5, 1), "P": (5, -1),
}
TELEOP_MAGNITUDE = 0.25


def key_to_action(key: int, glfw_module) -> np.ndarray | None:
    """Map one keypress to a small normalized Cartesian/gripper increment."""
    label = glfw_module.get_key_name(key, 0)
    if not label:
        return None
    binding = KEY_ACTIONS.get(label.upper())
    if binding is None:
        return None
    index, direction = binding
    action = np.zeros(6, dtype=np.float32)
    action[index] = direction * TELEOP_MAGNITUDE
    return action


def collect(stage: str, object_name: str | None, episodes: int, out: Path,
            append: bool = False, seed: int = 42):
    """Run the passive MuJoCo viewer and save every operator-labeled step."""
    import glfw

    env = SO101PickPlaceEnv(stage=stage, object_name=object_name)
    key_events: queue.Queue[int] = queue.Queue()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    names: list[str] = []
    episode_ids: list[int] = []
    episode_success: list[bool] = []
    current_obs = None
    current_name = object_name or "box"
    current_episode = 0
    episode_successful = False
    episode_steps = 0
    stopping = False

    def on_key(key):
        key_events.put(int(key))

    try:
        env.reset(seed=seed, options={"object_name": current_name})
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            print("W/S: X, A/D: Y, R/F: Z, J/L: wrist flex, I/U: wrist roll, O/P: gripper open/close")
            print("Each keypress makes one bounded 4.5 mm / 0.02 rad / 3% step. N ends an episode; Esc saves and quits.")
            deadline = time.monotonic()
            while viewer.is_running() and current_episode < episodes and not stopping:
                deadline += env.ctrl.settings.action_period_s
                try:
                    key = key_events.get_nowait()
                except queue.Empty:
                    key = None
                action = np.zeros(6, dtype=np.float32)
                finish_episode = False
                if key is not None:
                    if key == glfw.KEY_ESCAPE:
                        stopping = True
                    elif key in (glfw.KEY_N, glfw.KEY_ENTER):
                        finish_episode = True
                    else:
                        mapped = key_to_action(key, glfw)
                        if mapped is not None:
                            action += mapped
                if stopping:
                    break
                current_obs = env._observation()
                if np.any(action):
                    observations.append(current_obs.copy())
                    actions.append(action.copy())
                    names.append(current_name)
                    episode_ids.append(current_episode)
                _, _, terminated, truncated, info = env.step(action)
                episode_successful = bool(info["success"])
                episode_steps += 1
                viewer.sync()
                if terminated or truncated or finish_episode:
                    episode_success.append(episode_successful)
                    print(f"episode {current_episode + 1}: object={current_name}, steps={episode_steps}, success={episode_successful}")
                    current_episode += 1
                    episode_steps = 0
                    if current_episode >= episodes:
                        break
                    if object_name is None:
                        current_name = OBJECTS[(seed + current_episode) % len(OBJECTS)]
                    env.reset(seed=seed + current_episode, options={"object_name": current_name})
                    episode_successful = False
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if episode_steps:
            episode_success.append(episode_successful)
        env.close()

    if observations:
        count = save_demonstrations(out, observations, actions, names, episode_ids,
                                    episode_success, append=append)
        print(f"Saved {len(observations)} teleoperation steps ({count} total) to {out}")
    else:
        print("No teleoperation steps were recorded; no dataset written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("place", "mixed", "robust"), default="mixed")
    parser.add_argument("--object", choices=OBJECTS, help="lock every episode to one object; otherwise rotate objects")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/teleop/demonstrations.npz"))
    parser.add_argument("--append", action="store_true", help="append to an existing dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    collect(args.stage, args.object, args.episodes, args.out, args.append, args.seed)


if __name__ == "__main__":
    main()
