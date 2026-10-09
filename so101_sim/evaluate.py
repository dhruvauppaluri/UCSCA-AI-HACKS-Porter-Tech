"""Held-out policy evaluation and motion quality measurements."""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from .env import OBJECTS, SO101PickPlaceEnv


def evaluate_policy(model: SAC, stage: str = "robust", episodes_per_object: int = 5,
                    seed: int = 10000, video_dir: Path | None = None) -> dict:
    env = SO101PickPlaceEnv(stage=stage, render_mode="rgb_array" if video_dir else None)
    results = []
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio
    try:
        for object_name in OBJECTS:
            for episode in range(episodes_per_object):
                obs, info = env.reset(seed=seed + 100 * OBJECTS.index(object_name) + episode,
                                      options={"object_name": object_name})
                q_history = [np.array(info["joint_qpos"][:5])]
                frames = []
                success = False
                for _ in range(env.max_steps):
                    if video_dir and episode == 0:
                        frames.append(env.render())
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    q_history.append(np.array(info["joint_qpos"][:5]))
                    if terminated or truncated:
                        success = bool(info["success"])
                        break
                trajectory = np.stack(q_history)
                dt = env.ctrl.settings.action_period_s
                vel = np.diff(trajectory, axis=0) / dt
                accel = np.diff(vel, axis=0) / dt
                jerk = np.diff(accel, axis=0) / dt
                record = {
                    "object": object_name, "episode": episode, "success": success,
                    "grasped": bool(info["grasped_once"]), "lifted": bool(info["lifted_once"]),
                    "seconds": round(float(info["step_count"] * dt), 3),
                    "max_joint_speed_rad_s": float(np.max(np.abs(vel))) if vel.size else 0.0,
                    "rms_joint_accel_rad_s2": float(np.sqrt(np.mean(accel**2))) if accel.size else 0.0,
                    "rms_joint_jerk_rad_s3": float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
                    "collision_count": info["collision_count"],
                }
                results.append(record)
                if video_dir and episode == 0 and frames:
                    imageio.mimsave(video_dir / f"{object_name}.mp4", frames, fps=20)
    finally:
        env.close()
    summary = {name: {
        "success_rate": sum(r["success"] for r in results if r["object"] == name) / episodes_per_object,
        "grasp_rate": sum(r["grasped"] for r in results if r["object"] == name) / episodes_per_object,
        "mean_seconds": float(np.mean([r["seconds"] for r in results if r["object"] == name])),
    } for name in OBJECTS}
    return {"stage": stage, "episodes_per_object": episodes_per_object, "summary": summary, "episodes": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("reach", "grasp", "lift", "place", "mixed", "robust"), default="robust")
    parser.add_argument("--episodes-per-object", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    model = SAC.load(args.checkpoint, device="auto")
    result = evaluate_policy(model, args.stage, args.episodes_per_object, args.seed, args.video_dir)
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
