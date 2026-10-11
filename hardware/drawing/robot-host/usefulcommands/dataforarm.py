#!/usr/bin/env python3
"""Run SO-101 leader/follower teleoperation; press X to toggle CSV recording."""

import argparse
import csv
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig


def numeric_values(values: dict, prefix: str) -> dict[str, float]:
    """Keep only scalar numeric joints; skip camera image arrays."""
    result = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            result[f"{prefix}{key}"] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--teleop-port", default="/dev/ttyACM0")
    parser.add_argument("--teleop-id", default="my_awesome_leader_arm")
    parser.add_argument("--camera", default="/dev/video2", help="Camera device path; use 'none' to disable")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be greater than zero")
    if not sys.stdin.isatty():
        parser.error("Run this from an interactive terminal so X can be detected")

    output = args.out or Path.home() / f"arm_motion_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    cameras = {}
    if args.camera.lower() != "none":
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=Path(args.camera), fps=30, width=640, height=480, fourcc="YUYV"
        )

    follower = SOFollower(
        SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id, cameras=cameras)
    )
    leader = SOLeader(SOLeaderTeleopConfig(port=args.teleop_port, id=args.teleop_id))

    old_terminal = termios.tcgetattr(sys.stdin.fileno())
    follower_connected = False
    leader_connected = False
    recording = False
    writer = None
    file_handle = None
    period = 1.0 / args.hz
    start = time.monotonic()
    deadline = start

    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Starting teleoperation: leader /", args.teleop_port, "-> follower /", args.robot_port)
        print("Press X to start/stop CSV recording; press Q or Ctrl+C to exit.")
        leader.connect()
        leader_connected = True
        follower.connect()
        follower_connected = True

        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "x":
                    recording = not recording
                    if recording:
                        file_handle = output.open("a", newline="", encoding="utf-8")
                        print(f"Recording started: {output}")
                    else:
                        if file_handle:
                            file_handle.flush()
                            file_handle.close()
                            file_handle = None
                        writer = None
                        print("Recording paused/stopped. Press X to record again (appends new rows).")
                        # Keep the current CSV open in append mode for a later recording segment.
                        recording = False
                elif key == "q":
                    break

            observation = follower.get_observation()
            action = leader.get_action()
            sent_action = follower.send_action(action)

            if recording:
                leader_data = numeric_values(action, "leader_")
                follower_data = numeric_values(observation, "follower_")
                sent_data = numeric_values(sent_action or {}, "target_")
                row_data = {**leader_data, **follower_data, **sent_data}
                if writer is None:
                    if file_handle is None:
                        file_handle = output.open("a", newline="", encoding="utf-8")
                    fieldnames = ["time_s", *row_data.keys()]
                    writer = csv.DictWriter(file_handle, fieldnames=fieldnames, extrasaction="ignore")
                    if file_handle.tell() == 0:
                        writer.writeheader()
                    start = time.monotonic()
                writer.writerow({"time_s": f"{time.monotonic() - start:.6f}", **row_data})
                file_handle.flush()

            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopping teleoperation.")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal)
        if file_handle:
            file_handle.flush()
            file_handle.close()
        if follower_connected:
            follower.disconnect()
        if leader_connected:
            leader.disconnect()
        print(f"CSV output: {output}")


if __name__ == "__main__":
    main()
