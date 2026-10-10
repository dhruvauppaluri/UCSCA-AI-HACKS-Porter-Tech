"""Record human keyboard corrections with camera and joint actions in LeRobot v3."""

import argparse
import json
import queue
import time
from pathlib import Path

import cv2
import mujoco.viewer
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .camera import CameraEffects, MuJoCoCamera
from .control import ALL_JOINTS
from .env import OBJECTS, SO101PickPlaceEnv
from .teleop import key_to_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101-mujoco-keyboard")
    parser.add_argument("--episodes", type=int, default=10, help="successful episodes to save")
    parser.add_argument("--stage", choices=("mixed", "robust"), default="robust")
    parser.add_argument("--object", choices=OBJECTS)
    parser.add_argument("--seed", type=int, default=30000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"dataset output already exists: {args.out}")

    import glfw

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.images.overhead": {"dtype": "video", "shape": (120, 160, 3),
                                        "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(ALL_JOINTS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(ALL_JOINTS)},
    }
    dataset = LeRobotDataset.create(args.repo_id, fps=20, root=args.out,
                                    robot_type="so101_sim", features=features, use_videos=True)
    env = SO101PickPlaceEnv(stage=args.stage, render_mode="rgb_array")
    keys: queue.Queue[int] = queue.Queue()
    attempts = []
    saved = 0
    attempt = 0

    def on_key(key):
        keys.put(int(key))

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            print("W/S X, A/D Y, R/F Z, J/L wrist flex, I/U wrist roll, O/P gripper.")
            print("N or Enter ends an attempt; Esc saves completed episodes and quits.")
            while viewer.is_running() and saved < args.episodes:
                object_name = args.object or OBJECTS[attempt % len(OBJECTS)]
                seed = args.seed + attempt
                _, reset_info = env.reset(seed=seed, options={"object_name": object_name})
                camera = MuJoCoCamera(env, effects=CameraEffects(
                    jpeg_quality=60, brightness_range=0.15, blur_probability=0.1,
                    drop_probability=0, delay_frames=0, delay_jitter_frames=0,
                    perspective_jitter_pixels=2.0), seed=seed)
                frames = []
                info = reset_info
                deadline = time.monotonic()
                stop = False
                for _ in range(env.max_steps):
                    deadline += env.ctrl.settings.action_period_s
                    action = np.zeros(6, dtype=np.float32)
                    finish = False
                    while not keys.empty():
                        key = keys.get_nowait()
                        if key == glfw.KEY_ESCAPE:
                            stop = True
                            break
                        if key in (glfw.KEY_N, glfw.KEY_ENTER):
                            finish = True
                            break
                        mapped = key_to_action(key, glfw)
                        if mapped is not None:
                            action += mapped
                    if stop or finish:
                        break
                    rgb = cv2.resize(camera.get_frame().rgb, (160, 120), interpolation=cv2.INTER_AREA)
                    state = env._observation()[:6].copy()
                    joint_target = env.ctrl.proposed_target(env.data, action).astype(np.float32)
                    _, _, terminated, truncated, info = env.step(action)
                    frames.append((rgb, state, joint_target))
                    viewer.sync()
                    if terminated or truncated:
                        break
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                camera.close()
                success = bool(info["success"])
                attempts.append({"attempt": attempt, "seed": seed, "object": object_name,
                                 "success": success, "steps": info["step_count"],
                                 "randomization": reset_info["randomization"]})
                if success:
                    for rgb, state, action in frames:
                        dataset.add_frame({"observation.images.overhead": rgb,
                                           "observation.state": state, "action": action,
                                           "task": f"Pick up the {object_name} and place it in the tray"})
                    dataset.save_episode()
                    saved += 1
                print(f"attempt {attempt + 1}: {object_name}, success={success}, saved={saved}/{args.episodes}",
                      flush=True)
                attempt += 1
                if stop:
                    break
    finally:
        env.close()
        dataset.finalize()
        (args.out.parent / f"{args.out.name}-attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")


if __name__ == "__main__":
    main()
