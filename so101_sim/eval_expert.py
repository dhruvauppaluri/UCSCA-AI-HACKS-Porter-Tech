"""Measure the scripted teacher on fixed-seed held-out pick-and-place tasks."""

import argparse
import json
from pathlib import Path

import numpy as np

from .env import OBJECTS, SO101PickPlaceEnv
from .expert import ScriptedExpert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-object", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--stages", nargs="+", choices=("reach", "grasp", "lift", "place", "mixed", "robust"),
                        default=["mixed", "robust"])
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--physics-variant", choices=("original", "pick101_contact"),
                        default="original")
    parser.add_argument("--contact-height-offset-m", type=float, default=0.008)
    parser.add_argument("--contact-lateral-offset-m", type=float, default=0.0)
    args = parser.parse_args()
    results = []
    for stage_index, stage in enumerate(args.stages):
        env = SO101PickPlaceEnv(stage=stage, max_steps=args.max_steps,
                               assist_grasp=args.physics_variant != "pick101_contact",
                               physics_variant=args.physics_variant)
        try:
            for object_name in args.objects:
                object_index = OBJECTS.index(object_name)
                for episode in range(args.episodes_per_object):
                    seed = args.seed + stage_index * 10000 + object_index * 100 + episode
                    env.reset(seed=seed, options={"object_name": object_name})
                    expert = ScriptedExpert(env, contact_height_offset_m=args.contact_height_offset_m,
                                            contact_lateral_offset_m=args.contact_lateral_offset_m)
                    joint_history = [env.data.qpos[env.qids[:5]].copy()]
                    for _ in range(env.max_steps):
                        _, _, terminated, truncated, info = env.step(expert.act())
                        joint_history.append(env.data.qpos[env.qids[:5]].copy())
                        if terminated or truncated:
                            break
                    dt = env.ctrl.settings.action_period_s
                    speed = np.diff(np.stack(joint_history), axis=0) / dt
                    jerk = np.diff(np.diff(speed, axis=0) / dt, axis=0) / dt
                    results.append({
                        "stage": stage, "object": object_name, "episode": episode, "seed": seed,
                        "success": bool(info["success"]), "grasped": bool(info["grasped_once"]),
                        "lifted": bool(info["lifted_once"]), "seconds": float(info["step_count"] * dt),
                        "max_joint_speed_rad_s": float(np.max(np.abs(speed))) if speed.size else 0.0,
                        "rms_joint_jerk_rad_s3": float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
                        "collision_count": int(info["collision_count"]),
                        "randomization": info["randomization"],
                    })
                    print(f"{stage} {object_name} {episode + 1}/{args.episodes_per_object}: "
                          f"{'success' if info['success'] else 'failure'}", flush=True)
        finally:
            env.close()
    summary = {}
    for stage in args.stages:
        group = [r for r in results if r["stage"] == stage]
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
    args.out.write_text(json.dumps({"physics_variant": args.physics_variant,
                                    "max_steps": args.max_steps,
                                    "contact_height_offset_m": args.contact_height_offset_m,
                                    "contact_lateral_offset_m": args.contact_lateral_offset_m,
                                    "summary": summary, "episodes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
