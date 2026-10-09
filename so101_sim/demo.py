"""Typed simulated command, using the same perception contract as the API."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from stable_baselines3 import SAC

from .camera import CameraEffects, MuJoCoCamera
from .env import BASE_SIZES, OBJECTS, SO101PickPlaceEnv
from .vision import ColorObjectDetector, TableCalibration


class SimulationController:
    def __init__(self, checkpoint: Path, stage: str = "robust", seed: int = 42):
        self.env = SO101PickPlaceEnv(stage=stage, render_mode="rgb_array")
        self.policy = SAC.load(checkpoint, device="auto")
        self.camera = MuJoCoCamera(self.env, CameraEffects(), seed)
        self.detector = ColorObjectDetector(TableCalibration.simulated())
        self.obs, self.info = self.env.reset(seed=seed)
        self.state = "idle"
        self.stop_requested = False
        self.last_detections = []

    def list_objects(self):
        frame = self.camera.get_frame()
        detections = self.detector.detect(frame)
        self.last_detections = detections
        return [{"name": d.name, "confidence": d.confidence,
                 "pixel_center": d.pixel_center, "table_xy": d.table_xy,
                 "age_s": monotonic() - d.captured_at} for d in detections]

    def stop(self):
        self.stop_requested = True
        self.state = "stopped"
        return {"state": self.state}

    def status(self):
        return {"state": self.state, "object": self.info["object_name"],
                "step": self.info["step_count"], "success": self.info["success"],
                "grasped": self.info["grasped_once"], "lifted": self.info["lifted_once"]}

    def pick_and_place(self, object_name: str, destination: str = "tray", max_steps: int | None = None,
                       video_path: Path | None = None):
        if object_name not in OBJECTS:
            raise ValueError(f"object must be one of {OBJECTS}")
        if destination != "tray":
            raise ValueError("only the tray destination is modeled")
        self.obs, self.info = self.env.reset(options={"object_name": object_name})
        self.stop_requested = False
        self.state = "running"
        frames = []
        max_steps = max_steps or self.env.max_steps
        for step in range(max_steps):
            if self.stop_requested:
                break
            if step % 4 == 0:
                detections = self.list_objects()
                target = next((d for d in detections if d["name"] == object_name and d["age_s"] < 0.8), None)
                if target:
                    half_h = BASE_SIZES[object_name][2 if object_name == "box" else 1]
                    self.env.observed_object_xyz = np.array([*target["table_xy"], half_h])
                if video_path:
                    frames.append(self.env.render())
            self.obs = self.env._observation()
            action, _ = self.policy.predict(self.obs, deterministic=True)
            self.obs, _, done, truncated, self.info = self.env.step(action)
            if done or truncated:
                break
        if video_path and frames:
            import imageio.v2 as imageio
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=5)
        if self.state != "stopped":
            self.state = "succeeded" if self.info["success"] else "failed"
        return self.status()

    def close(self):
        self.camera.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--object", choices=OBJECTS, default="box")
    parser.add_argument("--stage", choices=("reach", "grasp", "lift", "place", "mixed", "robust"), default="robust")
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    controller = SimulationController(args.checkpoint, stage=args.stage)
    try:
        print("Visible objects:", json.dumps(controller.list_objects(), indent=2))
        print("Result:", json.dumps(controller.pick_and_place(args.object, video_path=args.video), indent=2))
    finally:
        controller.close()


if __name__ == "__main__":
    main()
