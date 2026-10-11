#!/usr/bin/env python3
"""Execute one phase of the calibrated 24 mm circle, using servo feedback only."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from fk_so101 import relative_height_mm

ROOT = Path.home() / "drawing_capture"
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PHASES = {"position": ("lift", "move_to_edge"),
          "descend": ("descend",), "trace": ("draw", "retract")}


def joints(bus):
    values = bus.sync_read("Present_Position")
    return {name: float(values[name]) for name in NAMES}


def gap(a, b):
    return max(abs(a[name]-b[name]) for name in NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()
    plan = json.loads((ROOT / "circle_plan.json").read_text())
    contact = plan["contact_joints_deg"]
    waypoints = [w for w in plan["waypoints"] if w["phase"] in PHASES[args.phase]]
    if not waypoints or max(w["ik_error_mm"] for w in waypoints) > .5:
        raise RuntimeError("invalid plan")
    robot = SO101Follower(SOFollowerRobotConfig(port=PORT, id="my_awesome_follower_arm"))
    bus = robot.bus
    bus.connect()  # Keep torque enabled; robot.connect() would release the arm briefly.
    log = []
    try:
        if not bus.is_calibrated:
            raise RuntimeError("calibration mismatch")
        torque = {name: bus.read("Torque_Enable", name) for name in bus.motors}
        if any(value != 1 for value in torque.values()):
            raise RuntimeError(f"torque disabled: {torque}")
        actual = joints(bus)
        expected = contact if args.phase == "position" else next(
            w["joints_deg"] for w in reversed(plan["waypoints"])
            if w["phase"] == ("move_to_edge" if args.phase == "descend" else "descend"))
        if gap(actual, expected) > 2:
            raise RuntimeError(f"start pose differs by {gap(actual,expected):.1f} degrees")
        print("phase", args.phase, "start", json.dumps(actual), flush=True)
        for index, waypoint in enumerate(waypoints, 1):
            goal = waypoint["joints_deg"]
            if gap(actual, goal) > 2:
                raise RuntimeError(f"step {index}: unexpected {gap(actual,goal):.1f} degree jump")
            bus.sync_write("Goal_Position", goal)
            deadline = time.monotonic()+3
            while True:
                time.sleep(.15)
                actual = joints(bus)
                height = relative_height_mm(actual, contact)
                log.append({"step":index,"phase":waypoint["phase"],"target":goal,
                            "actual":actual,"height_from_contact_mm":height})
                if args.phase == "position" and height < -1:
                    raise RuntimeError(f"tip fell {height:.1f} mm below contact height")
                if gap(actual,goal) <= .8:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"step {index}: tracking timeout; gap {gap(actual,goal):.1f} deg")
            if index % 10 == 0 or index == len(waypoints):
                print(f"{args.phase} {index}/{len(waypoints)} height {height:.1f} mm gap {gap(actual,goal):.1f} deg",flush=True)
        print("phase complete", args.phase, "final", json.dumps(actual),flush=True)
    finally:
        path = ROOT / "data" / f"circle_{args.phase}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps({"phase":args.phase,"log":log},indent=2)+"\n")
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
