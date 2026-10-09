"""Record diagnostic scripted pick-and-place rollouts for each object."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from .env import OBJECTS, SO101PickPlaceEnv
from .expert import ScriptedExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/expert_demo"))
    parser.add_argument("--stage", default="mixed", choices=("place", "mixed", "robust"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = SO101PickPlaceEnv(stage=args.stage, render_mode="rgb_array")
    results = []
    try:
        for index, name in enumerate(OBJECTS):
            env.reset(seed=args.seed + index, options={"object_name": name})
            expert = ScriptedExpert(env)
            frames = []
            for _ in range(env.max_steps):
                frames.append(env.render())
                _, _, terminated, truncated, info = env.step(expert.act())
                if terminated or truncated:
                    break
            frames.append(env.render())
            imageio.mimsave(args.out_dir / f"scripted_{name}.mp4", frames, fps=20)
            results.append({"object": name, "success": info["success"],
                            "steps": info["step_count"], "scripted_phase": expert.phase})
    finally:
        env.close()
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
