"""Bounded Cartesian-to-joint control used in training and deployment tests."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .gripper import GripperMapping


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_JOINTS = (*ARM_JOINTS, "gripper")


@dataclass
class ControlSettings:
    action_period_s: float = 0.05
    max_xyz_step_m: float = 0.018
    max_wrist_step_rad: float = 0.08
    max_gripper_step_percent: float = 12.0
    max_joint_step_rad: float = 0.065
    joint_margin_rad: float = 0.05
    ik_damping: float = 0.025


class SafeCartesianController:
    def __init__(self, model: mujoco.MjModel, settings: ControlSettings | None = None):
        self.model = model
        self.settings = settings or ControlSettings()
        self.site_id = model.site("gripperframe").id
        self.joint_ids = [model.joint(name).id for name in ALL_JOINTS]
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)
        self.limits = model.jnt_range[self.joint_ids].copy()
        self.target = np.zeros(6, dtype=float)
        self.gripper = GripperMapping()

    def reset(self, data: mujoco.MjData):
        self.target[:] = data.qpos[self.qpos_ids]
        data.ctrl[:] = self.target

    def proposed_target(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be six finite values")
        action = np.clip(action, -1, 1)
        self.target[:] = data.ctrl[:6]
        delta_xyz = action[:3] * self.settings.max_xyz_step_m
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        jac = jacp[:, self.dof_ids[:4]]
        gram = jac @ jac.T + np.eye(3) * self.settings.ik_damping**2
        dq = jac.T @ np.linalg.solve(gram, delta_xyz)
        candidate = self.target.copy()
        candidate[:4] += dq
        candidate[3] += action[3] * self.settings.max_wrist_step_rad
        candidate[4] += action[4] * self.settings.max_wrist_step_rad
        percent = self.gripper.to_percent(candidate[5])
        percent += action[5] * self.settings.max_gripper_step_percent
        candidate[5] = self.gripper.to_joint(percent)
        max_step = self.settings.max_joint_step_rad
        candidate[:5] = np.clip(candidate[:5], self.target[:5] - max_step, self.target[:5] + max_step)
        candidate = np.clip(
            candidate,
            self.limits[:, 0] + self.settings.joint_margin_rad * np.array([1, 1, 1, 1, 1, 0]),
            self.limits[:, 1] - self.settings.joint_margin_rad * np.array([1, 1, 1, 1, 1, 0]),
        )
        return candidate

    def apply(self, data: mujoco.MjData, action: np.ndarray, substeps: int):
        start = data.ctrl[:6].copy()
        end = self.proposed_target(data, action)
        return self._interpolate(data, start, end, substeps)

    def apply_joint_target(self, data: mujoco.MjData, target: np.ndarray, substeps: int):
        """Apply an absolute six-joint policy target with the same motion bounds."""
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("joint target must be six finite values")
        start = data.ctrl[:6].copy()
        end = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        max_step = self.settings.max_joint_step_rad
        end[:5] = np.clip(end[:5], start[:5] - max_step, start[:5] + max_step)
        gripper_percent = self.gripper.to_percent(start[5])
        requested_percent = self.gripper.to_percent(end[5])
        bounded_percent = np.clip(requested_percent, gripper_percent - self.settings.max_gripper_step_percent,
                                  gripper_percent + self.settings.max_gripper_step_percent)
        end[5] = self.gripper.to_joint(bounded_percent)
        return self._interpolate(data, start, end, substeps)

    def _interpolate(self, data: mujoco.MjData, start: np.ndarray, end: np.ndarray, substeps: int):
        for i in range(substeps):
            # Smoothstep interpolation has zero velocity at both endpoints.
            t = (i + 1) / substeps
            blend = t * t * (3 - 2 * t)
            data.ctrl[:6] = start + blend * (end - start)
            mujoco.mj_step(self.model, data)
        self.target[:] = end
        return end
