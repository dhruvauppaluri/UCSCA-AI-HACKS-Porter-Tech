"""Follower Z response with a motion-gated, position-dependent correction.

Inputs are leader/target XYZ and one initial follower Z reading. X and Y
predictions are never modified. No future follower measurements enter rollout.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "friday_voice_drawing_and_friction_model" / "data" / "recordings" / "cartesian"
NAMES = (
    "arm_motion_20261010_210241_cartesian.csv",
    "arm_motion_20261010_211211_cartesian.csv",
    "arm_motion_20261010_213515_cartesian.csv",
    "arm_motion_20261010_213829_cartesian.csv",
    "arm_motion_20261010_214539_cartesian.csv",
    "arm_motion_20261010_214924_cartesian.csv",
    "teleop_demo_20261011T035258Z_cartesian.csv",
)
DELAYS_S = np.arange(0, 0.167, 1 / 30)
TAUS_S = np.array([0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.21, 0.28, 0.38, 0.52])


def read_recording(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"Need at least two paired samples: {path}")
    time = np.array([float(r["time_s"]) for r in rows])
    target = np.array([[float(r[f"target_{axis}_mm"]) for axis in "xyz"]
                       for r in rows])
    actual = np.array([[float(r[f"follower_{axis}_mm"]) for axis in "xyz"]
                       for r in rows])
    if np.any(np.diff(time) <= 0) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(actual)):
        raise ValueError(f"Invalid timestamps or positions: {path}")
    return {"name": Path(path).name, "time": time, "target": target, "actual": actual}


def rollout(time_s, target_z_mm, initial_z_mm, delay_s, tau_s):
    if tau_s <= 0 or delay_s < 0:
        raise ValueError("Positive tau and nonnegative delay required")
    time = np.asarray(time_s, dtype=float)
    target = np.asarray(target_z_mm, dtype=float)
    command = np.interp(time - delay_s, time, target, left=target[0])
    predicted = np.empty_like(command)
    predicted[0] = initial_z_mm
    for index in range(1, len(time)):
        alpha = 1 - math.exp(-(time[index] - time[index - 1]) / tau_s)
        predicted[index] = predicted[index - 1] + alpha * (command[index] - predicted[index - 1])
    return predicted


def score(records, axis, delay_s, tau_s):
    return float(np.mean([
        np.mean((rollout(r["time"], r["target"][:, axis], r["actual"][0, axis],
                         delay_s, tau_s) - r["actual"][:, axis]) ** 2)
        for r in records
    ]))


RIDGE = 0.1
MOTION_SCALE_MM = 20.0
MOTION_DEADBAND_MM = 1.0
FEATURE_NAMES = ("intercept", "x", "y", "z", "x_squared", "y_squared",
                 "z_squared", "xy", "xz", "yz")


def motion_progress_mm(leader_xyz_mm):
    """Largest displacement from the start, with a small sensor deadband.

    The running maximum is causal and stays active after the arm returns.
    Summing every joint-to-joint Cartesian step amplifies stationary jitter.
    """
    leader = np.asarray(leader_xyz_mm, dtype=float)
    displacement = np.linalg.norm(leader - leader[0], axis=1)
    return np.maximum.accumulate(np.maximum(0., displacement - MOTION_DEADBAND_MM))


def features(leader_xyz_mm):
    leader = np.asarray(leader_xyz_mm, dtype=float)
    x = (leader[:, 0] - 250) / 100
    y = (leader[:, 1] + 100) / 100
    z = (leader[:, 2] - 20) / 50
    spatial = np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y,
                               z*z, x*y, x*z, y*z))
    gate = 1 - np.exp(-motion_progress_mm(leader) / MOTION_SCALE_MM)
    return spatial * gate[:, None]


def fit_response(records):
    return min(((score(records, 2, float(delay), float(tau)),
                 float(delay), float(tau))
                for delay in DELAYS_S for tau in TAUS_S),
               key=lambda item: item[0])[1:]


def baseline_z(time_s, leader_xyz_mm, initial_follower_z_mm, response):
    """Hold the measured initial offset until the leader begins moving."""
    time = np.asarray(time_s, dtype=float)
    leader = np.asarray(leader_xyz_mm, dtype=float)
    delay_s, tau_s = response
    base = rollout(time, leader[:, 2], initial_follower_z_mm, delay_s, tau_s)
    gate_remaining = np.exp(-motion_progress_mm(leader) / MOTION_SCALE_MM)
    initial_offset = initial_follower_z_mm - leader[0, 2]
    hold = initial_offset * (1 - np.exp(-(time - time[0]) / tau_s)) * gate_remaining
    return base + hold


def fit_model(records):
    if len(records) < 2:
        raise ValueError("Need at least two paired recordings")
    response = fit_response(records)
    matrix = np.concatenate([
        features(r["target"]) / np.sqrt(len(r["time"])) for r in records
    ])
    residual = np.concatenate([
        (r["actual"][:, 2] - baseline_z(
            r["time"], r["target"], r["actual"][0, 2], response)) /
        np.sqrt(len(r["time"])) for r in records
    ])
    penalty = np.eye(len(FEATURE_NAMES)) * RIDGE
    penalty[0, 0] = 0
    weights = np.linalg.solve(matrix.T @ matrix + penalty,
                              matrix.T @ residual)
    return {"delay_s": response[0], "tau_s": response[1],
            "ridge": RIDGE, "motion_scale_mm": MOTION_SCALE_MM,
            "motion_deadband_mm": MOTION_DEADBAND_MM,
            "motion_measure": "running maximum displacement from first leader XYZ",
            "feature_names": list(FEATURE_NAMES), "weights": weights.tolist()}


def predict_z(time_s, leader_xyz_mm, initial_follower_z_mm, model):
    if (model["feature_names"] != list(FEATURE_NAMES) or
            model["motion_scale_mm"] != MOTION_SCALE_MM or
            model.get("motion_deadband_mm") != MOTION_DEADBAND_MM or
            model.get("motion_measure") != "running maximum displacement from first leader XYZ"):
        raise ValueError("Unsupported Z predictor feature schema")
    return (baseline_z(time_s, leader_xyz_mm, initial_follower_z_mm,
                       (model["delay_s"], model["tau_s"])) +
            features(leader_xyz_mm) @ np.asarray(model["weights"]))
