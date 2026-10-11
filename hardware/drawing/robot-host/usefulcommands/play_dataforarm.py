#!/usr/bin/env python3
"""Replay a recorded SO-101 leader CSV as follower joint targets."""

import argparse
import csv
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


JOINT_COLUMNS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def load_trajectory(path: Path) -> tuple[list[tuple[float, dict[str, float]]], float]:
    """Read and validate the complete CSV before any hardware is connected."""
    rows: list[tuple[float, dict[str, float]]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        if all(f"target_{joint}" in fieldnames for joint in JOINT_COLUMNS):
            source_columns = {joint: f"target_{joint}" for joint in JOINT_COLUMNS}
        elif all(joint in fieldnames for joint in JOINT_COLUMNS):
            source_columns = {joint: joint for joint in JOINT_COLUMNS}
        else:
            missing = [
                f"target_{joint} (or {joint})"
                for joint in JOINT_COLUMNS
                if f"target_{joint}" not in fieldnames and joint not in fieldnames
            ]
            raise ValueError(f"CSV is missing required joint columns: {', '.join(missing)}")

        missing = {"time_s"} - fieldnames
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        previous_time = -1.0
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["time_s"])
                action = {joint: float(row[source]) for joint, source in source_columns.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc
            if timestamp < 0 or timestamp < previous_time:
                raise ValueError(f"time_s must be nonnegative and ordered (CSV line {line_number})")
            if not all(map(lambda value: value == value and abs(value) != float("inf"), action.values())):
                raise ValueError(f"Non-finite joint target on CSV line {line_number}")
            rows.append((timestamp, action))
            previous_time = timestamp

    if not rows:
        raise ValueError("CSV contains no motion rows")
    return rows, rows[-1][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="Recorded leader_movements_*.csv file")
    parser.add_argument("--robot-port", default="/dev/ttyACM1", help="Follower serial port")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm", help="Follower calibration ID")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Repeat the trajectory until Ctrl+C")
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero")

    trajectory, duration = load_trajectory(args.csv_file.expanduser())
    follower = SOFollower(SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id))
    connected = False
    try:
        print(f"Trajectory: {args.csv_file} ({len(trajectory)} points, {duration:.2f} s)")
        print(f"Follower: {args.robot_port} ({args.robot_id}); speed: {args.speed:g}x")
        print("Starting in 3 seconds. Keep clear of the arm and press Ctrl+C to stop.")
        time.sleep(3)
        follower.connect()
        connected = True

        while True:
            playback_start = time.monotonic()
            first_timestamp = trajectory[0][0]
            for timestamp, action in trajectory:
                target_elapsed = (timestamp - first_timestamp) / args.speed
                delay = playback_start + target_elapsed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                follower.send_action(action)
            if not args.loop:
                break
        print("Playback finished.")
    except KeyboardInterrupt:
        print("\nPlayback stopped.")
    finally:
        if connected:
            follower.disconnect()


if __name__ == "__main__":
    main()
