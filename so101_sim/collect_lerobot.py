"""Collect synchronized simulated SO-101 demonstrations in LeRobot format.

The scripted demonstrator uses privileged simulator state to label actions.
The saved policy inputs contain only an RGB camera frame and joint positions.
Joint actions are absolute MuJoCo radians, including the provisional gripper
joint, so they must be calibrated before use on a physical SO-101.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .camera import CameraEffects, MuJoCoCamera
from .control import ALL_JOINTS
from .env import OBJECTS, SO101PickPlaceEnv
from .expert import ScriptedExpert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Local LeRobot dataset directory")
    parser.add_argument("--repo-id", default="local/so101-mujoco-2026-10-09")
    parser.add_argument("--stage", choices=("grasp", "lift", "place", "mixed", "robust"), default="robust")
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument("--physics-variant", choices=("original", "pick101_contact"), default="original")
    parser.add_argument("--contact-height-offset-m", type=float, default=0.008)
    parser.add_argument("--contact-lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--successes-per-object", type=int, default=20)
    parser.add_argument("--max-attempts-per-object", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"dataset output already exists: {args.out}")

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.images.overhead": {
            "dtype": "video", "shape": (120, 160, 3), "names": ["height", "width", "channel"]},
        "observation.state": {
            "dtype": "float32", "shape": (6,), "names": list(ALL_JOINTS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(ALL_JOINTS)},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.out, fps=20,
                                    robot_type="so101_sim", features=features, use_videos=True)
    env = SO101PickPlaceEnv(stage=args.stage, render_mode="rgb_array",
                           assist_grasp=args.physics_variant != "pick101_contact",
                           physics_variant=args.physics_variant)
    records = []
    try:
        for object_name in args.objects:
            object_index = OBJECTS.index(object_name)
            successes = 0
            for attempt in range(args.max_attempts_per_object):
                if successes >= args.successes_per_object:
                    break
                seed = args.seed + object_index * 10000 + attempt
                _, reset_info = env.reset(seed=seed, options={"object_name": object_name})
                expert = ScriptedExpert(env, contact_height_offset_m=args.contact_height_offset_m,
                                        contact_lateral_offset_m=args.contact_lateral_offset_m)
                info = reset_info
                for _ in range(env.max_steps):
                    _, _, terminated, truncated, info = env.step(expert.act())
                    if terminated or truncated:
                        break
                succeeded = bool(info["success"])
                records.append({"object": object_name, "attempt": attempt, "seed": seed,
                                "stage": args.stage, "physics_variant": args.physics_variant,
                                "contact_height_offset_m": args.contact_height_offset_m,
                                "contact_lateral_offset_m": args.contact_lateral_offset_m,
                                "success": succeeded, "grasped": bool(info["grasped_once"]),
                                "lifted": bool(info["lifted_once"]), "steps": int(info["step_count"]),
                                "randomization": reset_info["randomization"]})
                if not succeeded:
                    continue
                # Failed attempts are cheap to screen without rendering. Replay
                # successful seeds to capture synchronized video, state, action.
                env.reset(seed=seed, options={"object_name": object_name})
                camera = MuJoCoCamera(env, effects=CameraEffects(
                    jpeg_quality=int(np.random.default_rng(seed).integers(38, 81)),
                    brightness_range=0.15, blur_probability=0.1, drop_probability=0,
                    delay_frames=0, delay_jitter_frames=0, perspective_jitter_pixels=2.0), seed=seed)
                expert = ScriptedExpert(env, contact_height_offset_m=args.contact_height_offset_m,
                                        contact_lateral_offset_m=args.contact_lateral_offset_m)
                frames = []
                try:
                    for _ in range(env.max_steps):
                        rgb = cv2.resize(camera.get_frame().rgb, (160, 120), interpolation=cv2.INTER_AREA)
                        state = env._observation()[:6].copy()
                        expert_action = expert.act()
                        intended_joint_target = env.ctrl.proposed_target(env.data, expert_action).astype(np.float32)
                        _, _, terminated, truncated, replay_info = env.step(expert_action)
                        frames.append((rgb, state, intended_joint_target))
                        if terminated or truncated:
                            break
                finally:
                    camera.close()
                if not replay_info["success"]:
                    raise RuntimeError(f"successful seed {seed} failed deterministic video replay")
                for rgb, state, action in frames:
                    dataset.add_frame({"observation.images.overhead": rgb,
                                       "observation.state": state, "action": action,
                                       "task": f"Pick up the {object_name} and place it in the tray"})
                dataset.save_episode()
                successes += 1
                print(f"{object_name}: {successes}/{args.successes_per_object} saved ({attempt + 1} attempts)", flush=True)
            if successes < args.successes_per_object:
                print(f"{object_name}: collected {successes}; attempt limit reached", flush=True)
    finally:
        env.close()
        dataset.finalize()
        (args.out.parent / f"{args.out.name}-attempts.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
