"""Fit joint response across all paired recordings and validate by held-out run.

Model: q[k] = q[k-1] + a * (target(t[k] - delay) - q[k-1]),
where a = 1-exp(-(t[k]-t[k-1])/tau). This is a useful motion predictor,
not a motor torque, contact, or friction model.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "recordings" / "raw"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
DELAYS_S = np.arange(0, 0.167, 1 / 30)
TAUS_S = np.array([0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.21, 0.28, 0.38, 0.52])


def read_csv(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty recording: {path}")
    time = np.array([float(r["time_s"]) for r in rows])
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"Non-increasing time: {path}")
    if "target_shoulder_pan.pos" not in rows[0]:
        return {"name": path.name, "kind": "leader_only", "time": time,
                "leader": np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in rows])}
    return {"name": path.name, "kind": "paired", "time": time,
            "target": np.array([[float(r[f"target_{j}.pos"]) for j in JOINTS] for r in rows]),
            "follower": np.array([[float(r[f"follower_{j}.pos"]) for j in JOINTS] for r in rows]),
            "leader": np.array([[float(r[f"leader_{j}.pos"]) for j in JOINTS] for r in rows])}


def read_jsonl(path):
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    time = np.array([float(r["elapsed_s"]) for r in rows])
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"Non-increasing time: {path}")
    return {"name": path.name, "kind": "paired", "time": time,
            "target": np.array([[float(r["sent_action"][f"{j}.pos"]) for j in JOINTS] for r in rows]),
            "follower": np.array([[float(r["follower_observation"][f"{j}.pos"]) for j in JOINTS] for r in rows]),
            "leader": np.array([[float(r["leader_action"][f"{j}.pos"]) for j in JOINTS] for r in rows])}


def load_recordings():
    csvs = [read_csv(p) for p in sorted(ARCHIVE.rglob("*.csv"))]
    full = ROOT / "data" / "recordings" / "session" / "teleop_demo_20261011T035258Z.jsonl"
    # The three-circle snapshot is a prefix of this full recording; omit it.
    return csvs + [read_jsonl(full)]


def rollout(time, target, initial, delay_s, tau_s):
    if tau_s <= 0 or delay_s < 0:
        raise ValueError("Positive tau and nonnegative delay required")
    t = np.asarray(time, dtype=float)
    target = np.asarray(target, dtype=float)
    t_delayed = t - delay_s
    command = np.interp(t_delayed, t, target, left=target[0])
    predicted = np.empty_like(command)
    predicted[0] = initial
    for k in range(1, len(t)):
        a = 1 - math.exp(-(t[k] - t[k - 1]) / tau_s)
        predicted[k] = predicted[k - 1] + a * (command[k] - predicted[k - 1])
    return predicted


def score(records, joint_index, delay_s, tau_s):
    errors = []
    for r in records:
        actual = r["follower"][:, joint_index]
        predicted = rollout(r["time"], r["target"][:, joint_index],
                            actual[0], delay_s, tau_s)
        errors.append(np.mean((predicted - actual) ** 2))
    return float(np.mean(errors))  # each independent run gets equal weight


def fit(records):
    return {joint: min(((score(records, index, float(delay), float(tau)), float(delay), float(tau))
                        for delay in DELAYS_S for tau in TAUS_S), key=lambda x: x[0])[1:]
            for index, joint in enumerate(JOINTS)}


def evaluate(r, params):
    target, actual, t = r["target"], r["follower"], r["time"]
    prediction = np.column_stack([
        rollout(t, target[:, i], actual[0, i], *params[j])
        for i, j in enumerate(JOINTS)
    ])
    # Gripper opening is not in angular degrees; keep it out of the arm error.
    arm = slice(0, 5)
    rms = lambda a: float(np.sqrt(np.mean(np.sum(a[:, arm] ** 2, axis=1))))
    return {"model_arm_rmse_deg": round(rms(prediction - actual), 3),
            "direct_target_arm_rmse_deg": round(rms(target - actual), 3),
            "model_joint_rmse_deg": {j: round(float(np.sqrt(np.mean((prediction[:, i]-actual[:, i])**2))), 3)
                                     for i, j in enumerate(JOINTS)}}


def leader_summary(r):
    q = r["leader"]
    rates = np.abs(np.diff(q, axis=0) / np.diff(r["time"])[:, None])
    return {"recording": r["name"], "samples": len(r["time"]),
            "duration_s": round(float(r["time"][-1]-r["time"][0]), 2),
            "joint_range_deg": {j: round(float(np.ptp(q[:, i])), 2)
                                for i, j in enumerate(JOINTS[:-1])},
            "p95_angular_speed_deg_s": {j: round(float(np.percentile(rates[:, i], 95)), 2)
                                          for i, j in enumerate(JOINTS[:-1])}}


def main():
    records = load_recordings()
    paired = [r for r in records if r["kind"] == "paired"]
    solo = [r for r in records if r["kind"] == "leader_only"]
    if len(paired) < 2:
        raise ValueError("At least two independent paired runs required")
    full = fit(paired)
    held_out = []
    for r in paired:
        trained = fit([other for other in paired if other is not r])
        held_out.append({"recording": r["name"], "samples": len(r["time"]),
                         "duration_s": round(float(r["time"][-1] - r["time"][0]), 2),
                         **evaluate(r, trained)})
    output = {
        "model": "independent joint first-order response with pure command delay",
        "units": "arm joints in degrees; gripper in its recorded opening units",
        "fitted_on": [r["name"] for r in paired],
        "leader_only": [leader_summary(r) for r in solo],
        "excluded_duplicate": "three_circles snapshot is contained in the full teleoperation recording",
        "parameters": {j: {"delay_s": round(d, 6), "tau_s": round(tau, 6)}
                       for j, (d, tau) in full.items()},
        "leave_one_run_out": held_out,
        "limits": ["The model describes observed tracking, not rigid-body dynamics or measured forces.",
                   "Contact state is not verified in these files; friction cannot be isolated from residual motion.",
                   "Cartesian pen coordinates inherit the URDF and pen-tip calibration error."],
    }
    path = ROOT / "analysis" / "all_recordings_model.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
