#!/usr/bin/env python3
"""Slowly move from the observed clear pose to the first hover-square waypoint."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from fk_so101 import relative_height_mm


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EXPECTED = {"shoulder_pan": 32.1758241758, "shoulder_lift": 21.8901098901,
            "elbow_flex": -9.6703296703, "wrist_flex": 87.2087912088,
            "wrist_roll": 18.5934065934}
PLAN = Path.home() / "drawing_capture" / "hover_square_plan.json"
OUTPUT = Path.home() / "drawing_capture" / "data" / "safe_approach.json"


def gap(a, b):
    return max(abs(a[name] - b[name]) for name in NAMES)


def main():
    goal = json.loads(PLAN.read_text())["waypoints"][0]["joints_deg"]
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id="my_awesome_follower_arm"))
    bus = robot.bus
    bus.connect()  # Do not run robot.configure(): it briefly releases torque.
    log = []
    try:
        if not bus.is_calibrated:
            raise RuntimeError("Motor calibration mismatch")
        torque = {name: bus.read("Torque_Enable", name) for name in bus.motors}
        if any(value != 1 for value in torque.values()):
            raise RuntimeError(f"Torque must already be enabled: {torque}")
        observed = bus.sync_read("Present_Position")
        initial = {name: float(observed[name]) for name in NAMES}
        print("initial", json.dumps(initial), flush=True)
        if gap(initial, EXPECTED) > 3:
            raise RuntimeError("Arm changed since clearance was confirmed")
        steps = math.ceil(gap(initial, goal) / 0.25)
        for index in range(1, steps + 1):
            f = index / steps
            target = {name: initial[name]*(1-f) + goal[name]*f for name in NAMES}
            bus.sync_write("Goal_Position", target)
            time.sleep(0.1)
            observed = bus.sync_read("Present_Position")
            actual = {name: float(observed[name]) for name in NAMES}
            error = gap(actual, target)
            height = relative_height_mm(actual, initial)
            log.append({"step": index, "target": target, "observed": actual,
                        "tracking_gap_deg": error, "rise_from_clear_pose_mm": height})
            if error > 10:
                raise RuntimeError(f"Tracking gap {error:.1f} deg; stopped")
            if height < -2:
                raise RuntimeError(f"Observed pose fell {height:.1f} mm below clear pose; stopped")
            if index % 20 == 0 or index == steps:
                print(f"approach step {index}/{steps}; rise {height:.1f} mm; tracking gap {error:.1f} deg", flush=True)
        time.sleep(0.5)
        observed = bus.sync_read("Present_Position")
        actual = {name: float(observed[name]) for name in NAMES}
        print("approach complete", json.dumps(actual), flush=True)
    finally:
        OUTPUT.write_text(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                      "log": log}, indent=2) + "\n")
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
