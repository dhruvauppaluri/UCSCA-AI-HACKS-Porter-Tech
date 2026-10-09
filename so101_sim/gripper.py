"""Explicit LeRobot percent-to-MuJoCo joint mapping.

The upstream XML does not encode which physical jaw direction corresponds to
LeRobot's 0=closed, 100=open. This provisional direction must be checked on
the loaner arm before commanding its gripper.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperMapping:
    closed_rad: float = -0.1745329776
    open_rad: float = 1.7453291996

    def to_joint(self, percent_open: float) -> float:
        fraction = min(1.0, max(0.0, percent_open / 100.0))
        return self.closed_rad + fraction * (self.open_rad - self.closed_rad)

    def to_percent(self, joint_rad: float) -> float:
        span = self.open_rad - self.closed_rad
        return 100.0 * min(1.0, max(0.0, (joint_rad - self.closed_rad) / span))
