#!/usr/bin/env python3
"""Run the prechecked, pen-up center square on the SO-101 follower.

Only run after the live teleoperation process has released the follower port.
This script does not draw; it returns to its starting pose afterward.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from fk_so101 import relative_height_mm


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAMERA_STREAM = "http://127.0.0.1:8767/cam0"
CLEAR_POSE = {
    "shoulder_pan": 32.175824175824175,
    "shoulder_lift": 21.89010989010989,
    "elbow_flex": -9.67032967032967,
    "wrist_flex": 87.20879120879121,
    "wrist_roll": 18.593406593406595,
}


def joint_dict(observation):
    return {name: float(observation[name]) for name in NAMES}


def max_gap(a, b):
    return max(abs(a[name] - b[name]) for name in NAMES)


def move(bus, goal, log, max_step_deg=0.25, interval_s=0.1):
    start = joint_dict(bus.sync_read("Present_Position"))
    steps = max(1, math.ceil(max_gap(start, goal) / max_step_deg))
    for step in range(1, steps + 1):
        f = step / steps
        target = {name: start[name] * (1 - f) + goal[name] * f for name in NAMES}
        bus.sync_write("Goal_Position", target)
        time.sleep(interval_s)
        observed = joint_dict(bus.sync_read("Present_Position"))
        tracking_gap = max_gap(observed, target)
        rise = relative_height_mm(observed, CLEAR_POSE)
        log.append({"at_utc": datetime.now(timezone.utc).isoformat(), "target": target,
                    "observed": observed, "tracking_gap_deg": tracking_gap,
                    "rise_from_clear_pose_mm": rise})
        if tracking_gap > 10:
            raise RuntimeError(f"Stopped: joint tracking gap {tracking_gap:.1f} deg exceeds 10 deg")
        if rise < -2:
            raise RuntimeError(f"Stopped: observed pose fell {rise:.1f} mm below clear pose")
    time.sleep(0.3)
    final_gap = max_gap(joint_dict(bus.sync_read("Present_Position")), goal)
    if final_gap > 8:
        raise RuntimeError(f"Stopped: final joint gap {final_gap:.1f} deg exceeds 8 deg")


def snapshot(label, output_dir):
    cap = cv2.VideoCapture(CAMERA_STREAM)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("No cam0 frame")
        path = output_dir / f"{label}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not save {path}")
        return str(path)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "drawing_capture" / "data" / "hover_square")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    waypoints = plan["waypoints"]
    if plan["side_mm"] != 100 or plan["height_mm"] != 45 or len(waypoints) != 81:
        raise RuntimeError("Unexpected hover-square plan")
    if max(item["ik_error_mm"] for item in waypoints) > 1:
        raise RuntimeError("Plan contains IK error over 1 mm")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = []
    frames = {}
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id="my_awesome_follower_arm"))
    bus = robot.bus
    bus.connect()  # Keep torque on; robot.connect() temporarily disables it.
    try:
        if not bus.is_calibrated:
            raise RuntimeError("Follower calibration does not match")
        torque = {name: bus.read("Torque_Enable", name) for name in bus.motors}
        if any(value != 1 for value in torque.values()):
            raise RuntimeError(f"Torque must already be enabled: {torque}")
        start = joint_dict(bus.sync_read("Present_Position"))
        print("starting_joints", json.dumps(start), flush=True)
        if max_gap(start, waypoints[0]["joints_deg"]) > 10:
            raise RuntimeError("Arm is too far from the first square waypoint")
        if relative_height_mm(start, CLEAR_POSE) < -2:
            raise RuntimeError("Arm is below the user-confirmed clear pose")
        frames["before"] = snapshot("before", args.output_dir)
        move(bus, waypoints[0]["joints_deg"], log)
        for i, item in enumerate(waypoints[1:], 1):
            move(bus, item["joints_deg"], log)
            if i in (20, 40, 60, 80):
                label = f"corner_{i // 20}"
                frames[label] = snapshot(label, args.output_dir)
                print(f"reached {label}", flush=True)
        move(bus, CLEAR_POSE, log)
        frames["after"] = snapshot("after", args.output_dir)
        print("hover square complete; returned to the user-confirmed clear pose", flush=True)
    finally:
        result = {"at_utc": datetime.now(timezone.utc).isoformat(), "plan": str(args.plan),
                  "frames": frames, "motion_log": log}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
