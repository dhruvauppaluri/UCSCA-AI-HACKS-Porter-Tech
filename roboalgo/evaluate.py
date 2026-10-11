"""Compare predicted follower Z with a fresh paired Cartesian recording."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from z_model import NAMES, predict_z, read_recording


def summarize(error):
    absolute = np.abs(error)
    return {
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "mean_signed_error_mm": float(np.mean(error)),
        "p95_absolute_error_mm": float(np.percentile(absolute, 95)),
        "max_absolute_error_mm": float(np.max(absolute)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Paired Cartesian CSV with target_{x,y,z}_mm and follower_{x,y,z}_mm")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "z_predictor_all_seven.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = read_recording(args.input)
    model = json.loads(args.model.read_text())
    actual = run["actual"][:, 2]
    predicted = predict_z(run["time"], run["target"], actual[0], model)
    error = predicted - actual
    first_two = run["time"] - run["time"][0] < 2
    report = {
        "input": str(args.input),
        "model": str(args.model),
        "samples": len(error),
        "duration_s": float(run["time"][-1] - run["time"][0]),
        "training_recording": run["name"] in NAMES,
        "initial_follower_z_mm": float(actual[0]),
        "z_prediction_vs_recorded": summarize(error),
        "first_2s": summarize(error[first_two]),
        "direct_target_z_vs_recorded": summarize(run["target"][:, 2] - actual),
        "interpretation": "Follower Z is forward-kinematics derived in robot base frame; it is not directly measured pen-paper clearance or pressure.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time_s", "target_x_mm", "target_y_mm", "target_z_mm",
                         "predicted_follower_z_mm", "recorded_follower_z_mm", "z_error_mm"))
        for time, target, estimate, observed, delta in zip(
                run["time"], run["target"], predicted, actual, error):
            writer.writerow((time, *target, estimate, observed, delta))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
