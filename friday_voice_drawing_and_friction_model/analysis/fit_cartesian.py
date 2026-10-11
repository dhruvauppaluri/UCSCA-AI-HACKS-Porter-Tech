"""Fit a direct XYZ response proxy for simulation of planned pen paths."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fit_all_recordings import DELAYS_S, TAUS_S, rollout

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "recordings" / "cartesian"
NAMES = (
    "arm_motion_20261010_210241_cartesian.csv",
    "arm_motion_20261010_211211_cartesian.csv",
    "arm_motion_20261010_213515_cartesian.csv",
    "arm_motion_20261010_213829_cartesian.csv",
    "arm_motion_20261010_214539_cartesian.csv",
    "arm_motion_20261010_214924_cartesian.csv",
    "teleop_demo_20261011T035258Z_cartesian.csv",
)


def read(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {"name": path.name,
            "time": np.array([float(r["time_s"]) for r in rows]),
            "target": np.array([[float(r[f"target_{a}_mm"]) for a in "xyz"] for r in rows]),
            "actual": np.array([[float(r[f"follower_{a}_mm"]) for a in "xyz"] for r in rows])}


def score(records, axis, delay, tau):
    return float(np.mean([
        np.mean((rollout(r["time"], r["target"][:, axis], r["actual"][0, axis],
                         delay, tau) - r["actual"][:, axis]) ** 2)
        for r in records
    ]))


def fit(records):
    return {a: min(((score(records, i, float(d), float(t)), float(d), float(t))
                   for d in DELAYS_S for t in TAUS_S), key=lambda v: v[0])[1:]
            for i, a in enumerate("xyz")}


def fit_isotropic(records):
    """One response for all axes, so it can be used in paper coordinates."""
    return min(((float(np.mean([score(records, i, float(d), float(t)) for i in range(3)])),
                 float(d), float(t)) for d in DELAYS_S for t in TAUS_S),
               key=lambda v: v[0])[1:]


def evaluate(r, params):
    predicted = np.column_stack([
        rollout(r["time"], r["target"][:, i], r["actual"][0, i], *params[a])
        for i, a in enumerate("xyz")
    ])
    rmse = lambda e: float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    return {"model_xyz_rmse_mm": round(rmse(predicted-r["actual"]), 3),
            "direct_target_xyz_rmse_mm": round(rmse(r["target"]-r["actual"]), 3)}


def main():
    runs = [read(SOURCE/name) for name in NAMES]
    full = fit(runs)
    isotropic = fit_isotropic(runs)
    checks = []
    for r in runs:
        params = fit([other for other in runs if other is not r])
        iso = fit_isotropic([other for other in runs if other is not r])
        check = evaluate(r, params)
        iso_check = evaluate(r, {axis: iso for axis in "xyz"})
        checks.append({"recording": r["name"], "samples": len(r["time"]), **check,
                       "isotropic_model_xyz_rmse_mm": iso_check["model_xyz_rmse_mm"]})
    result = {"model": "Independent XYZ delayed first-order response in URDF base frame",
              "source": "Seven independent paired runs; leader-only runs have no follower XYZ",
              "parameters": {axis: {"delay_s": round(d, 6), "tau_s": round(t, 6)}
                             for axis, (d, t) in full.items()},
              "isotropic_parameters": {"delay_s": round(isotropic[0], 6),
                                       "tau_s": round(isotropic[1], 6)},
              "leave_one_run_out": checks,
              "limits": "XYZ values are derived from imperfect forward kinematics; paper contact and friction are not identified."}
    (ROOT/"analysis"/"all_cartesian_model.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
