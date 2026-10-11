#!/usr/bin/env python3
"""SO-101 leader/follower teleoperation with on-demand corner snapshots.

Commands:
  python corner_capture.py run
  python corner_capture.py mark bottom_left
  python corner_capture.py record-start circle_01
  python corner_capture.py record-stop
  python corner_capture.py status

The `mark` command never opens a serial port. It requests a snapshot from the
running control loop, which saves the follower's calibrated joint positions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C123878-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00"
LEADER_ID = "my_awesome_leader_arm"
FOLLOWER_ID = "my_awesome_follower_arm"
LABELS = ("bottom_left", "bottom_right", "top_left", "top_right")
PIVOT_LABELS = tuple(f"pivot_{i}" for i in range(1, 7))
DEFAULT_DATA_DIR = Path.home() / "drawing_capture" / "data"
RECORD_ACTIONS = ("start", "stop")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def port_holders(ports: tuple[str, str]) -> list[tuple[int, str]]:
    """Find processes with either serial device open, before connecting."""
    devices = {os.path.realpath(port) for port in ports}
    holders = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.path.realpath(fd)
                except OSError:
                    continue
                if target in devices:
                    holders.append((int(proc.name), target))
                    break
        except (OSError, PermissionError):
            continue
    return holders


def make_hardware():
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    leader = SO101Leader(SOLeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))
    follower = SO101Follower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    for arm in (leader, follower):
        if len(arm.calibration) != 6:
            raise RuntimeError(f"Missing six-motor calibration: {arm.calibration_fpath}")
    processors = make_default_processors()
    return leader, follower, processors


class SimulatedArm:
    is_connected = True
    is_calibrated = True

    def __init__(self):
        self.count = 0

    def connect(self, calibrate=False):
        return None

    def disconnect(self):
        return None

    def get_observation(self):
        self.count += 1
        return {f"{name}.pos": float(self.count + i) for i, name in enumerate(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        )}

    def get_action(self):
        return self.get_observation()

    def send_action(self, action):
        return action


def save_pending(data_dir: Path, observation: dict[str, float]) -> str | None:
    requests = data_dir / "requests"
    last_label = None
    for request_path in sorted(requests.glob("*.json")):
        request = read_json(request_path, {})
        label = request.get("label")
        request_id = request.get("id")
        if label not in (*LABELS, *PIVOT_LABELS) or not request_id or request_path.stem != request_id:
            request_path.unlink(missing_ok=True)
            continue
        captures_path = data_dir / ("corners.json" if label in LABELS else "pivot.json")
        positions = {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
        if len(positions) != 6:
            raise RuntimeError(f"Expected six follower joint positions, found {len(positions)}")
        capture = {
            "label": label,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "follower_id": FOLLOWER_ID,
            "follower_port": FOLLOWER_PORT,
            "joint_positions": positions,
            "units": {"body_joints": "degrees", "gripper": "percent_open"},
            "pen_tip_offset_mm_estimate": 38.1,
            "note": "Pen offset direction and exact length are not yet calibrated.",
        }
        all_captures = read_json(captures_path, {})
        all_captures[label] = capture
        atomic_json(captures_path, all_captures)
        atomic_json(data_dir / "acks" / f"{request_id}.json", capture)
        request_path.unlink(missing_ok=True)
        print(f"Captured {label}: {positions}", flush=True)
        last_label = label
    return last_label


def handle_record_requests(data_dir: Path, recording: dict[str, object]) -> None:
    """Apply record start/stop requests inside the owning teleop loop."""
    requests = data_dir / "record_requests"
    requests.mkdir(parents=True, exist_ok=True)
    for request_path in sorted(requests.glob("*.json")):
        request = read_json(request_path, {})
        request_id = request.get("id")
        action = request.get("action")
        label = request.get("label", "teleop")
        if not request_id or request_path.stem != request_id or action not in RECORD_ACTIONS:
            request_path.unlink(missing_ok=True)
            continue
        result: dict[str, object] = {"id": request_id, "action": action}
        if action == "start":
            if recording.get("stream") is not None:
                result["error"] = f"Recording already active: {recording.get('label')}"
            else:
                safe_label = "".join(ch for ch in str(label) if ch.isalnum() or ch in "-_")[:48]
                if not safe_label:
                    result["error"] = "Recording label must contain letters or numbers"
                else:
                    path = data_dir / f"{safe_label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
                    recording["stream"] = path.open("a", encoding="utf-8", buffering=1)
                    recording["path"] = str(path)
                    recording["label"] = safe_label
                    recording["count"] = 0
                    recording["started_at_utc"] = datetime.now(timezone.utc).isoformat()
                    recording["started_monotonic"] = time.monotonic()
                    result["path"] = str(path)
        else:
            stream = recording.get("stream")
            if stream is None:
                result["error"] = "No recording is active"
            else:
                stream.flush()
                stream.close()
                result.update({"path": recording.get("path"), "samples": recording.get("count")})
                recording.update({"stream": None, "path": None, "label": None,
                                  "count": 0, "started_at_utc": None,
                                  "started_monotonic": None})
        atomic_json(data_dir / "record_acks" / f"{request_id}.json", result)
        request_path.unlink(missing_ok=True)
        print(json.dumps(result), flush=True)


def run(args: argparse.Namespace) -> None:
    def stop_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    data_dir = args.data_dir
    (data_dir / "requests").mkdir(parents=True, exist_ok=True)
    (data_dir / "acks").mkdir(parents=True, exist_ok=True)
    (data_dir / "record_requests").mkdir(parents=True, exist_ok=True)
    (data_dir / "record_acks").mkdir(parents=True, exist_ok=True)
    status_path = data_dir / "status.json"

    if args.simulate:
        leader = SimulatedArm()
        follower = SimulatedArm()
        processors = None
    else:
        for port in (LEADER_PORT, FOLLOWER_PORT):
            if not Path(port).exists():
                raise RuntimeError(f"Serial device is missing: {port}")
        if os.path.realpath(LEADER_PORT) == os.path.realpath(FOLLOWER_PORT):
            raise RuntimeError("Leader and follower ports resolve to the same device")
        holders = port_holders((LEADER_PORT, FOLLOWER_PORT))
        if holders:
            raise RuntimeError(f"Serial ports already in use: {holders}")
        leader, follower, processors = make_hardware()

    try:
        leader.connect(calibrate=False)
        follower.bus.connect()
        if not leader.is_calibrated or not follower.is_calibrated:
            raise RuntimeError("Motor calibration mismatch; capture mode will not recalibrate automatically")
        # Power-on Goal_Position can be far from Present_Position. Set a safe
        # hold target before configure() enables torque.
        follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
        follower.configure()
        print(f"Capture teleoperation ready; data directory: {data_dir}", flush=True)
        last_status = 0.0
        last_label = None
        recording: dict[str, object] = {"stream": None, "path": None, "label": None,
                                        "count": 0, "started_monotonic": None}
        interval = 1.0 / args.fps
        while True:
            start = time.monotonic()
            observation = follower.get_observation()
            handle_record_requests(data_dir, recording)
            captured = save_pending(data_dir, observation)
            if captured:
                last_label = captured
            action = leader.get_action()
            stream = recording.get("stream")
            if stream is not None:
                sample = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.monotonic() - float(recording["started_monotonic"]),
                    "follower_observation": {key: float(value) for key, value in observation.items()
                                             if key.endswith(".pos")},
                    "leader_action": {key: float(value) for key, value in action.items()
                                      if key.endswith(".pos")},
                }
                try:
                    stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    recording["count"] = int(recording["count"]) + 1
                except OSError as exc:
                    print(f"Recording stopped after file error: {exc}", flush=True)
                    stream.close()
                    recording.update({"stream": None, "path": None, "label": None,
                                      "count": 0, "started_at_utc": None,
                                      "started_monotonic": None})
            if processors is not None:
                teleop_processor, robot_processor, _ = processors
                action = teleop_processor((action, observation))
                action = robot_processor((action, observation))
            # Avoid a jump when the leader and follower start at different poses.
            for key, value in tuple(action.items()):
                if key.endswith(".pos") and key in observation:
                    present = float(observation[key])
                    action[key] = max(present - 0.3, min(present + 0.3, float(value)))
            follower.send_action(action)
            if time.monotonic() - last_status >= 0.5:
                atomic_json(status_path, {
                    "running": True,
                    "simulated": args.simulate,
                    "pid": os.getpid(),
                    "heartbeat_unix": time.time(),
                    "last_capture": last_label,
                    "capture_count": len(read_json(data_dir / "corners.json", {})),
                    "recording": recording.get("label"),
                    "recording_samples": recording.get("count", 0),
                })
                last_status = time.monotonic()
            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("Stopping capture teleoperation", flush=True)
    finally:
        stream = locals().get("recording", {}).get("stream")
        if stream is not None:
            stream.flush()
            stream.close()
        atomic_json(status_path, {"running": False, "pid": os.getpid(), "heartbeat_unix": time.time()})
        if follower.is_connected:
            follower.bus.disconnect(disable_torque=False)
        if leader.is_connected:
            leader.disconnect()


def mark(args: argparse.Namespace) -> None:
    data_dir = args.data_dir
    status = read_json(data_dir / "status.json", {})
    if not status.get("running") or time.time() - status.get("heartbeat_unix", 0) > 2:
        raise RuntimeError("Capture teleoperation is not running")
    request_id = uuid.uuid4().hex
    request_path = data_dir / "requests" / f"{request_id}.json"
    ack_path = data_dir / "acks" / f"{request_id}.json"
    atomic_json(request_path, {"id": request_id, "label": args.label})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if ack_path.exists():
            print(json.dumps(read_json(ack_path, {}), indent=2))
            ack_path.unlink(missing_ok=True)
            return
        time.sleep(0.05)
    request_path.unlink(missing_ok=True)
    raise RuntimeError("Capture timed out; hold the pose and retry")


def status(args: argparse.Namespace) -> None:
    state = read_json(args.data_dir / "status.json", {})
    state["heartbeat_fresh"] = bool(state.get("running") and time.time() - state.get("heartbeat_unix", 0) <= 2)
    state["corners"] = list(read_json(args.data_dir / "corners.json", {}).keys())
    state["pivot_samples"] = list(read_json(args.data_dir / "pivot.json", {}).keys())
    print(json.dumps(state, indent=2))


def record(args: argparse.Namespace) -> None:
    data_dir = args.data_dir
    status = read_json(data_dir / "status.json", {})
    if not status.get("running") or time.time() - status.get("heartbeat_unix", 0) > 2:
        raise RuntimeError("Capture teleoperation is not running")
    request_id = uuid.uuid4().hex
    request_path = data_dir / "record_requests" / f"{request_id}.json"
    ack_path = data_dir / "record_acks" / f"{request_id}.json"
    atomic_json(request_path, {"id": request_id, "action": args.action, "label": args.label})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if ack_path.exists():
            result = read_json(ack_path, {})
            ack_path.unlink(missing_ok=True)
            print(json.dumps(result, indent=2))
            if result.get("error"):
                raise RuntimeError(result["error"])
            return
        time.sleep(0.05)
    request_path.unlink(missing_ok=True)
    raise RuntimeError("Recording request timed out; teleoperation can remain running")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Start leader-to-follower teleoperation with capture requests")
    run_parser.add_argument("--fps", type=int, default=60)
    run_parser.add_argument("--simulate", action="store_true", help="Test capture without serial devices")
    mark_parser = commands.add_parser("mark", help="Capture the follower's current joint positions")
    mark_parser.add_argument("label", choices=(*LABELS, *PIVOT_LABELS))
    start_parser = commands.add_parser("record-start", help="Begin recording teleop joint samples without restarting")
    start_parser.add_argument("label", nargs="?", default="teleop")
    commands.add_parser("record-stop", help="Finish the active joint sample recording; teleoperation keeps running")
    commands.add_parser("status", help="Show capture service status and saved corner labels")
    args = parser.parse_args()
    if args.command == "run":
        if args.fps < 1 or args.fps > 120:
            parser.error("--fps must be between 1 and 120")
        run(args)
    elif args.command == "mark":
        mark(args)
    elif args.command == "record-start":
        args.action = "start"
        record(args)
    elif args.command == "record-stop":
        args.action = "stop"
        args.label = None
        record(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
