"""Validate the motion-gated Z predictor by run, then fit all paired runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from z_model import NAMES, SOURCE, fit_model, predict_z, read_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_recording(SOURCE / name) for name in NAMES]
    validation = []
    for held_out in runs:
        training = [r for r in runs if r is not held_out]
        model = fit_model(training)
        predicted = predict_z(held_out["time"], held_out["target"],
                              held_out["actual"][0, 2], model)
        error = predicted - held_out["actual"][:, 2]
        first_two = held_out["time"] - held_out["time"][0] < 2
        validation.append({
            "recording": held_out["name"], "samples": len(error),
            "z_rmse_mm": float(np.sqrt(np.mean(error**2))),
            "first_2s_z_rmse_mm": float(np.sqrt(np.mean(error[first_two]**2))),
            "mean_z_error_mm": float(np.mean(error)),
        })
        if held_out["name"] == "arm_motion_20261010_213515_cartesian.csv":
            with (args.output_dir / "held_out_213515_z.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("time_s", "leader_z_mm", "predicted_z_mm",
                                 "recorded_follower_z_mm", "error_mm"))
                for time, leader, estimate, actual, delta in zip(
                        held_out["time"], held_out["target"][:, 2], predicted,
                        held_out["actual"][:, 2], error):
                    writer.writerow((time, leader, estimate, actual, delta))
    final_model = fit_model(runs)
    final_model.update({
        "model": "motion-gated spatial correction of delayed follower Z",
        "training_recordings": list(NAMES),
        "trained_on_all_paired_runs": True,
        "input": "leader/target XYZ in base-frame mm plus initial follower Z in mm",
        "output": "predicted follower Z in base-frame mm",
        "x_y_predictions": "unchanged; this model returns Z only",
        "limitations": [
            "Leader-only and duplicate-prefix files have no independent paired follower labels.",
            "Model form was developed after inspecting these runs; leave-one-run-out scores are diagnostics, not new-session validation.",
            "Base-frame XYZ comes from forward kinematics, not directly measured pen-tip or paper contact.",
        ],
    })
    (args.output_dir / "z_predictor_all_seven.json").write_text(
        json.dumps(final_model, indent=2) + "\n")
    report = {
        "validation_method": "leave one paired run out of parameter fitting",
        "mean_run_z_rmse_mm": float(np.mean([r["z_rmse_mm"] for r in validation])),
        "runs": validation,
        "final_model": "z_predictor_all_seven.json",
    }
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
