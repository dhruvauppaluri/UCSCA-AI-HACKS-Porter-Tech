"""Diagnostic finite-state demonstrator for curriculum bootstrapping.

It uses simulator state and the assisted grasp. It is a source of trajectories,
not a deployable robot controller or evidence of learned performance.
"""

import numpy as np


class ScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.phase = "approach"
        self.lift_target = None
        self.close_target = None

    def act(self) -> np.ndarray:
        env = self.env
        obj = env._object_xyz()
        ee = env._ee_xyz()
        tray = env._tray_xyz()
        if self.phase == "approach" and np.linalg.norm(ee[:2] - obj[:2]) < 0.018 and abs(ee[2] - obj[2]) < 0.055:
            self.phase = "close"
            self.close_target = ee.copy() if env.selected == "bottle" else obj + np.array([0, 0, 0.025])
        if self.phase == "close" and env._assisted_attached:
            self.phase = "lift"
            self.lift_target = ee + np.array([0, 0, 0.10])
        if self.phase == "lift" and obj[2] > env._object_half_height() + 0.055:
            self.phase = "move"
        if self.phase == "move" and np.linalg.norm(ee[:2] - tray[:2]) < 0.018:
            self.phase = "lower"
        lower_target = tray + np.array([0, 0, 0.035])
        if self.phase == "lower" and np.linalg.norm(ee - lower_target) < 0.015:
            self.phase = "release"
        if self.phase == "release" and env.gripper_map.to_percent(env.data.qpos[env.qids[5]]) > 82:
            self.phase = "settle"
        targets = {
            "approach": obj + np.array([0, 0, 0.025]),
            "close": self.close_target if self.close_target is not None else ee,
            "lift": self.lift_target if self.lift_target is not None else ee,
            "move": tray + np.array([0, 0, 0.10]),
            "lower": lower_target,
            "release": lower_target,
            "settle": lower_target,
        }
        action = np.zeros(6, dtype=np.float32)
        action[:3] = np.clip((targets[self.phase] - ee) / env.ctrl.settings.max_xyz_step_m, -1, 1)
        action[5] = -1 if self.phase in ("close", "lift", "move", "lower") else 1 if self.phase == "release" else 0
        return action
