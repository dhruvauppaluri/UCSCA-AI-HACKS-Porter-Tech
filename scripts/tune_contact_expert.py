"""Search a small scripted contact-grasp grid on tuning seeds only.

Use a disjoint seed range with eval_expert for the final score. This script
does not train a policy; it identifies a demonstrator for data collection.
"""

import argparse
import json
from pathlib import Path

from so101_sim.env import OBJECTS, SO101PickPlaceEnv
from so101_sim.expert import ScriptedExpert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, default="cylinder")
    parser.add_argument("--stage", choices=("grasp", "lift", "place", "mixed", "robust"), default="robust")
    parser.add_argument("--seed", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    trials = []
    env = SO101PickPlaceEnv(stage=args.stage, object_name=args.object,
                           assist_grasp=False, physics_variant="pick101_contact")
    try:
        for height in (0.0, 0.008, 0.016):
            for lateral in (0.0, 0.006, 0.012):
                results = []
                for episode in range(args.episodes):
                    seed = args.seed + OBJECTS.index(args.object) * 100 + episode
                    env.reset(seed=seed)
                    expert = ScriptedExpert(env, contact_height_offset_m=height,
                                            contact_lateral_offset_m=lateral)
                    for _ in range(env.max_steps):
                        _, _, terminated, truncated, info = env.step(expert.act())
                        if terminated or truncated:
                            break
                    results.append({"seed": seed, "grasped": bool(info["grasped_once"]),
                                    "lifted": bool(info["lifted_once"]), "placed": bool(info["success"]),
                                    "steps": info["step_count"]})
                row = {"height_m": height, "lateral_m": lateral, "pick": sum(x["grasped"] for x in results),
                       "lift": sum(x["lifted"] for x in results), "place": sum(x["placed"] for x in results),
                       "episodes": results}
                trials.append(row)
                print(f"height={height:.3f} lateral={lateral:.3f} "
                      f"pick={row['pick']} lift={row['lift']} place={row['place']}/{args.episodes}", flush=True)
    finally:
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"object": args.object, "stage": args.stage,
                                    "seed_base": args.seed, "trials": trials}, indent=2) + "\n")


if __name__ == "__main__":
    main()
