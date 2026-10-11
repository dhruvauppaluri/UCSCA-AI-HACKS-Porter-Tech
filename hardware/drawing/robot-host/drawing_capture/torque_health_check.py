#!/usr/bin/env python3
"""Read-only preflight health check for both SO-101 arms.

Reads Torque_Enable, Status, Present_Load, Present_Current, and Present_Position for every
motor on both buses and flags the failure signatures this project has already hit once:
a single stranded follower motor (torque off, no fault latched -- the shoulder_lift incident),
a latched hardware fault (Status != 0), a follower joint far outside its expected range, or the
leader ending up torqued when it should stay passive. Never writes any register.

Run this before any motion script. Exit code is 0 when everything looks healthy, 1 otherwise,
so callers (humans or a future voice-triggered orchestrator) can gate on it without parsing JSON.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C123878-if00"

# From hover_square.py's CLEAR_POSE -- the follower's confirmed-safe resting pose. Gripper is
# excluded: its range is percent-open, not degrees, and isn't part of the gravity-sag failure mode.
CLEAR_POSE_DEG = {
    "shoulder_pan": 32.175824175824175,
    "shoulder_lift": 21.89010989010989,
    "elbow_flex": -9.67032967032967,
    "wrist_flex": 87.20879120879121,
    "wrist_roll": 18.593406593406595,
}
POSITION_ENVELOPE_DEG = 60.0  # wide enough for real work poses, tight enough to catch a ~127deg sag

DATA_DIR = Path.home() / "drawing_capture" / "data"


def read_all(bus):
    return {
        motor: {
            "torque_enable": int(bus.read("Torque_Enable", motor)),
            "status": int(bus.read("Status", motor)),
            "present_load": int(bus.read("Present_Load", motor)),
            "present_current": int(bus.read("Present_Current", motor)),
            "present_position": float(bus.read("Present_Position", motor)),
        }
        for motor in bus.motors
    }


@dataclass
class HealthReport:
    follower: dict
    leader: dict
    latched_fault_motors: list = field(default_factory=list)
    stranded_motors: list = field(default_factory=list)
    out_of_range_motors: list = field(default_factory=list)
    leader_torqued_motors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.latched_fault_motors
            or self.stranded_motors
            or self.out_of_range_motors
            or self.leader_torqued_motors
        )

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "latched_fault_motors": self.latched_fault_motors,
            "stranded_motors": self.stranded_motors,
            "out_of_range_motors": self.out_of_range_motors,
            "leader_torqued_motors": self.leader_torqued_motors,
            "follower": self.follower,
            "leader": self.leader,
        }


def evaluate(follower_readings: dict, leader_readings: dict) -> HealthReport:
    report = HealthReport(follower=follower_readings, leader=leader_readings)

    for motor, reading in follower_readings.items():
        if reading["status"] != 0:
            report.latched_fault_motors.append({"motor": motor, "status": reading["status"]})
            continue  # a latched fault explains anything else wrong with this motor; don't double-report
        if reading["torque_enable"] != 1:
            report.stranded_motors.append({"motor": motor, "torque_enable": reading["torque_enable"]})
        if motor in CLEAR_POSE_DEG:
            gap = abs(reading["present_position"] - CLEAR_POSE_DEG[motor])
            if gap > POSITION_ENVELOPE_DEG:
                report.out_of_range_motors.append(
                    {"motor": motor, "present_position": reading["present_position"], "gap_deg": gap}
                )

    for motor, reading in leader_readings.items():
        if reading["status"] != 0:
            report.latched_fault_motors.append({"motor": f"leader/{motor}", "status": reading["status"]})
            continue
        if reading["torque_enable"] != 0:
            report.leader_torqued_motors.append({"motor": motor, "torque_enable": reading["torque_enable"]})

    return report


def run_check() -> HealthReport:
    follower = SO101Follower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id="my_awesome_follower_arm"))
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id="my_awesome_leader_arm"))

    readings = {}
    for label, bus in (("follower", follower.bus), ("leader", leader.bus)):
        bus.connect()
        try:
            readings[label] = read_all(bus)
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)

    return evaluate(readings["follower"], readings["leader"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="If exactly one follower motor is stranded (torque off, Status==0) and nothing else "
        "is wrong, invoke recover_stranded_motor.py for that motor. Refuses for any broader problem.",
    )
    args = parser.parse_args()

    report = run_check()
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {"at_utc": timestamp, **report.to_dict()}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.replace(":", "").replace("+00:00", "Z")
    out_path = DATA_DIR / f"health_check_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Health check @ {timestamp}  ok={report.ok}")
        if report.latched_fault_motors:
            print(f"  LATCHED FAULT (Status != 0, needs a power-cycle, never auto-recovered): "
                  f"{report.latched_fault_motors}")
        if report.stranded_motors:
            print(f"  STRANDED (torque off, no latched fault): {report.stranded_motors}")
        if report.out_of_range_motors:
            print(f"  OUT OF RANGE: {report.out_of_range_motors}")
        if report.leader_torqued_motors:
            print(f"  LEADER TORQUED (should be off): {report.leader_torqued_motors}")
        print(f"  Log written to {out_path}")

    if not report.ok and args.auto_recover:
        single_stranded_motor_only = (
            not report.latched_fault_motors
            and not report.out_of_range_motors
            and not report.leader_torqued_motors
            and len(report.stranded_motors) == 1
        )
        if single_stranded_motor_only:
            motor = report.stranded_motors[0]["motor"]
            print(f"--auto-recover: invoking recover_stranded_motor.py --motor {motor}")
            script = Path(__file__).with_name("recover_stranded_motor.py")
            return subprocess.call([sys.executable, str(script), "--motor", motor])
        print("--auto-recover: refusing -- more than a single stranded motor is wrong; a human must look at this.")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
