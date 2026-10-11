#!/usr/bin/env python3
"""Supervised follower-only 21 placement check, 30 mm nominal pen-up.

The arm must already be within 2 degrees of the first waypoint. This runner
never descends to the paper and never sends a gripper command.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from fk_so101 import tip_position
from corner_capture import FOLLOWER_PORT, FOLLOWER_ID, port_holders


NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DEFAULT_PLAN = (Path(__file__).resolve().parents[4] / "roboalgo" / "examples" / "21" /
                "physical_hover_21_FOR_REVIEW.csv")


def read_plan(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 265:
        raise ValueError("Expected exactly 265 hover rows")
    previous = -1.0
    result = []
    for row in rows:
        t = float(row["time_s"])
        goal = {name: float(row[f"{name}.pos"]) for name in NAMES}
        if (t <= previous or not all(math.isfinite(v) for v in goal.values()) or
                int(row["pen_down"]) != 0 or abs(float(row["paper_z_mm"]) - 30.0) > 0.01 or
                float(row["fk_residual_mm"]) > 1.0):
            raise ValueError("Invalid hover waypoint")
        result.append((t, goal))
        previous = t
    if abs(result[0][0]) > 1e-6 or result[-1][0] > 30:
        raise ValueError("Unexpected timing")
    return result


def gap(a, b):
    return max(abs(a[n] - b[n]) for n in NAMES)


def measured(bus):
    reading = bus.sync_read("Present_Position")
    return {n: float(reading[n]) for n in NAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--log", type=Path, default=Path.home() / "drawing_capture" / "data" /
                        f"hover_21_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl")
    parser.add_argument("--speed", type=float, default=0.4, help="Fraction of planned trajectory speed; default 0.4")
    args = parser.parse_args()
    if not 0 < args.speed <= 0.5:
        parser.error("speed must be >0 and <=0.5")
    plan = read_plan(args.plan)
    if port_holders((FOLLOWER_PORT,)):
        raise RuntimeError("Follower serial port is already owned")
    robot = SO101Follower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    bus = robot.bus
    args.log.parent.mkdir(parents=True, exist_ok=True)
    bus.connect()  # robot.connect/configure can momentarily release torque.
    try:
        if not bus.is_calibrated:
            raise RuntimeError("Follower calibration mismatch")
        torque = {name: bus.read("Torque_Enable", name) for name in bus.motors}
        if any(value != 1 for value in torque.values()):
            raise RuntimeError(f"Torque must already be enabled: {torque}")
        start = measured(bus)
        if gap(start, plan[0][1]) > 2.0:
            raise RuntimeError("Start pose more than 2 degrees from first hover target; position via teleoperation first")
        reference = plan[0][1]
        normal = (0.07559753628499707, -0.02460366430479077, 0.9968348269449736)
        def modeled_height(q):
            point = tip_position(q) - tip_position(reference)
            return 30.0 + 1000.0 * sum(a*b for a, b in zip(point, normal))
        if modeled_height(start) < 25:
            raise RuntimeError("Start pose below 25 mm model hover floor")
        with args.log.open("x", buffering=1) as stream:
            begin = time.monotonic()
            for index, (t, goal) in enumerate(plan):
                wait = begin + t / args.speed - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                actual = measured(bus)
                tracking = gap(actual, goal)
                height = modeled_height(actual)
                if tracking > 3.0 or height < 20.0:
                    raise RuntimeError(f"Stopped at {index}: gap {tracking:.2f} deg, modeled height {height:.1f} mm")
                bus.sync_write("Goal_Position", goal)
                stream.write(json.dumps({"time_s": t, "index": index, "target": goal,
                                         "observed": actual, "tracking_gap_deg": tracking,
                                         "modeled_paper_height_mm": height}) + "\n")
                if index % 30 == 0:
                    print(f"hover {index}/{len(plan)-1}: gap={tracking:.2f} deg, model_height={height:.1f} mm", flush=True)
            print(f"Hover complete; log {args.log}", flush=True)
    finally:
        bus.disconnect(disable_torque=False)  # Keep last position held.


if __name__ == "__main__":
    main()
