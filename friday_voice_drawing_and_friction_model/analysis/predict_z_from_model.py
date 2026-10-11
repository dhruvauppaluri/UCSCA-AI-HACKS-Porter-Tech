"""Apply the all-seven Z model to a new leader Cartesian CSV.

The caller supplies a measured initial follower Z; no later follower values
are read. X and Y are not predicted or changed by this command.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from z_motion_predictor import predict_z


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True,
                   help="CSV with time_s and leader_model_{x,y,z}_mm")
    p.add_argument("--initial-follower-z-mm", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("Need at least two leader samples")
    t = np.array([float(r["time_s"]) for r in rows])
    xyz = np.array([[float(r[f"leader_model_{a}_mm"]) for a in "xyz"]
                    for r in rows])
    if (np.any(np.diff(t) <= 0) or not np.all(np.isfinite(xyz)) or
            not np.isfinite(args.initial_follower_z_mm)):
        raise ValueError("Input must contain increasing times and finite positions")
    model = json.loads(args.model.read_text())
    estimate = predict_z(t, xyz, args.initial_follower_z_mm, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "leader_x_mm", "leader_y_mm", "leader_z_mm",
                         "predicted_follower_z_mm"))
        for time, lead, z in zip(t, xyz, estimate):
            writer.writerow((time, *lead, z))
    print(f"Saved {len(estimate)} Z predictions to {args.output}")


if __name__ == "__main__":
    main()
