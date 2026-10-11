#!/usr/bin/env python3
"""Lift the follower pen slowly without disabling torque during connection."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EXPECTED = {
    "shoulder_pan": 32.2637362637,
    "shoulder_lift": 27.5164835165,
    "elbow_flex": -10.1978021978,
    "wrist_flex": 79.9120879121,
    "wrist_roll": 18.9450549451,
}
# Numerical IK for a 40 mm rise along the fitted paper normal. Checked offline
# for monotonic upward tip motion throughout the joint interpolation.
LIFTED = {
    "shoulder_pan": 31.92,
    "shoulder_lift": 17.03,
    "elbow_flex": -12.05,
    "wrist_flex": 87.92,
    "wrist_roll": 19.06,
}
OUTPUT = Path.home() / "drawing_capture" / "data" / "slow_lift.json"


def gap(a, b):
    return max(abs(a[name] - b[name]) for name in NAMES)


def main():
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id="my_awesome_follower_arm"))
    bus = robot.bus
    bus.connect()  # Deliberately skip robot.connect(), which disables torque in configure().
    log = []
    try:
        if not bus.is_calibrated:
            raise RuntimeError("Motor calibration mismatch")
        torque = {name: bus.read("Torque_Enable", name) for name in bus.motors}
        if any(value != 1 for value in torque.values()):
            raise RuntimeError(f"Torque must already be enabled: {torque}")
        initial_all = bus.sync_read("Present_Position")
        initial = {name: float(initial_all[name]) for name in NAMES}
        print("initial", json.dumps(initial), flush=True)
        if gap(initial, EXPECTED) > 3:
            raise RuntimeError("Current pose differs from verified starting pose")
        steps = math.ceil(gap(initial, LIFTED) / 0.25)
        for index in range(1, steps + 1):
            fraction = index / steps
            target = {name: initial[name] * (1-fraction) + LIFTED[name] * fraction for name in NAMES}
            bus.sync_write("Goal_Position", target)
            time.sleep(0.1)
            observed_all = bus.sync_read("Present_Position")
            observed = {name: float(observed_all[name]) for name in NAMES}
            tracking_gap = gap(observed, target)
            log.append({"step": index, "target": target, "observed": observed,
                        "tracking_gap_deg": tracking_gap})
            if tracking_gap > 8:
                raise RuntimeError(f"Tracking gap {tracking_gap:.1f} deg; stopped")
            if index % 10 == 0 or index == steps:
                print(f"lift step {index}/{steps}; tracking gap {tracking_gap:.1f} deg", flush=True)
        time.sleep(0.5)
        final_all = bus.sync_read("Present_Position")
        final = {name: float(final_all[name]) for name in NAMES}
        if gap(final, LIFTED) > 3:
            raise RuntimeError(f"Final pose tracking gap {gap(final, LIFTED):.1f} deg")
        print("lift complete", json.dumps(final), flush=True)
    finally:
        OUTPUT.write_text(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                      "log": log}, indent=2) + "\n")
        bus.disconnect(disable_torque=False)  # Keep the raised pose held.


if __name__ == "__main__":
    main()
