"""Evaluate a saved LeRobot ACT policy with held-out MuJoCo episodes."""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame

from .camera import CameraEffects, MuJoCoCamera
from .control import ALL_JOINTS
from .env import OBJECTS, SO101PickPlaceEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--episodes-per-object", type=int, default=5)
    parser.add_argument("--stages", nargs="+", choices=("mixed", "robust"), default=["mixed", "robust"])
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physics-variant", choices=("original", "pick101_contact"),
                        default="original")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset, video_backend="pyav")
    policy = ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}})
    results = []
    videos_saved = set()
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for stage_index, stage in enumerate(args.stages):
            env = SO101PickPlaceEnv(
                stage=stage, render_mode="rgb_array", control_mode="joint",
                assist_grasp=args.physics_variant != "pick101_contact",
                physics_variant=args.physics_variant)
            try:
                for object_index, object_name in enumerate(OBJECTS):
                    for episode in range(args.episodes_per_object):
                        seed = args.seed + stage_index * 10000 + object_index * 100 + episode
                        env.reset(seed=seed, options={"object_name": object_name})
                        camera = MuJoCoCamera(env, effects=CameraEffects(
                            jpeg_quality=int(np.random.default_rng(seed).integers(38, 81)),
                            brightness_range=0.15, blur_probability=0.1,
                            drop_probability=0, delay_frames=0, delay_jitter_frames=0,
                            perspective_jitter_pixels=2.0), seed=seed)
                        policy.reset()
                        joint_history = [env.data.qpos[env.qids[:5]].copy()]
                        frames = []
                        info = None
                        with torch.inference_mode():
                            for _ in range(env.max_steps):
                                rgb = cv2.resize(camera.get_frame().rgb, (160, 120),
                                                 interpolation=cv2.INTER_AREA)
                                if args.video_dir:
                                    frames.append(rgb)
                                raw_obs = dict(zip(ALL_JOINTS, env._observation()[:6].copy()))
                                raw_obs["overhead"] = rgb
                                frame = build_inference_frame(raw_obs, torch.device(args.device),
                                                              dataset.meta.features)
                                action = postprocessor(policy.select_action(preprocessor(frame)))
                                joint_target = action.squeeze(0).cpu().numpy()
                                _, _, terminated, truncated, info = env.step(joint_target)
                                joint_history.append(env.data.qpos[env.qids[:5]].copy())
                                if terminated or truncated:
                                    break
                        camera.close()
                        joint_history = np.stack(joint_history)
                        dt = env.ctrl.settings.action_period_s
                        speed = np.diff(joint_history, axis=0) / dt
                        accel = np.diff(speed, axis=0) / dt
                        jerk = np.diff(accel, axis=0) / dt
                        record = {
                            "stage": stage, "object": object_name, "episode": episode, "seed": seed,
                            "success": bool(info["success"]), "grasped": bool(info["grasped_once"]),
                            "lifted": bool(info["lifted_once"]), "seconds": float(info["step_count"] * dt),
                            "max_joint_speed_rad_s": float(np.max(np.abs(speed))) if speed.size else 0.0,
                            "rms_joint_jerk_rad_s3": float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
                            "collision_count": int(info["collision_count"]),
                            "randomization": info["randomization"],
                        }
                        results.append(record)
                        outcome = "success" if record["success"] else "failure"
                        video_key = (stage, object_name, outcome)
                        if args.video_dir and video_key not in videos_saved:
                            imageio.mimsave(args.video_dir / f"{stage}_{object_name}_{outcome}.mp4", frames, fps=20)
                            videos_saved.add(video_key)
                        print(f"{stage} {object_name} {episode + 1}/{args.episodes_per_object}: {outcome}", flush=True)
            finally:
                env.close()
    finally:
        summary = {}
        for stage in args.stages:
            group = [item for item in results if item["stage"] == stage]
            if group:
                summary[stage] = {"episodes": len(group),
                                  "pick_rate": float(np.mean([r["grasped"] for r in group])),
                                  "lift_rate": float(np.mean([r["lifted"] for r in group])),
                                  "place_rate": float(np.mean([r["success"] for r in group])),
                                  "failures": sum(not r["success"] for r in group),
                                  "mean_seconds": float(np.mean([r["seconds"] for r in group])),
                                  "mean_max_joint_speed_rad_s": float(np.mean([
                                      r["max_joint_speed_rad_s"] for r in group])),
                                  "mean_rms_joint_jerk_rad_s3": float(np.mean([
                                      r["rms_joint_jerk_rad_s3"] for r in group]))}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"dataset": str(args.dataset), "checkpoint": str(args.checkpoint),
                                        "physics_variant": args.physics_variant,
                                        "summary": summary, "episodes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
