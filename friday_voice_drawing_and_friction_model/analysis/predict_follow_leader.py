"""Predict a held-out follower XYZ trajectory from leader XYZ samples.

Fit the existing delayed first-order Cartesian model on every *other* paired
recording. The evaluated follower run supplies only its initial XYZ state;
subsequent follower samples are used for scoring, never fitting or rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_all_recordings import DELAYS_S, TAUS_S, rollout
from fit_cartesian import NAMES, SOURCE, fit, read, score


FIELDS = (
    "source_row", "time_s", "leader_x_mm", "leader_y_mm", "leader_z_mm",
    "predicted_follower_x_mm", "predicted_follower_y_mm",
    "predicted_follower_z_mm", "recorded_follower_x_mm",
    "recorded_follower_y_mm", "recorded_follower_z_mm", "error_3d_mm",
    "predicted_follower_z_corrected_mm", "corrected_error_3d_mm",
)


def read_pasted(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("Need at least two rows")
    t = np.array([float(r["time_s"]) for r in rows])
    leader = np.array([[float(r[f"leader_model_{a}_mm"]) for a in "xyz"]
                       for r in rows])
    target = np.array([[float(r[f"target_{a}_mm"]) for a in "xyz"]
                       for r in rows])
    follower = np.array([[float(r[f"follower_{a}_mm"]) for a in "xyz"]
                         for r in rows])
    if np.any(np.diff(t) <= 0) or not all(np.all(np.isfinite(a)) for a in (leader, target, follower)):
        raise ValueError("Times must increase and all positions must be finite")
    if not np.allclose(leader, target, atol=1e-5, rtol=0):
        raise ValueError("Leader and sent target differ; this predictor needs the sent target")
    return rows, t, leader, follower


def z_features(time, leader, kind):
    """Causal spatial features; the first follower sample remains the anchor."""
    x = (leader[:, 0] - 250) / 100
    y = (leader[:, 1] + 100) / 100
    z = (leader[:, 2] - 20) / 50
    features = [np.ones(len(time))]
    if kind in ("linear_xyz", "quadratic_xyz"):
        features.extend((x, y, z))
    if kind == "quadratic_xyz":
        features.extend((x*x, y*y, z*z, x*y, x*z, y*z))
    ramp = 1 - np.exp(-(time - time[0]) / .2)
    return np.column_stack(features) * ramp[:, None]


def fit_z_response(records):
    return min(((score(records, 2, float(delay), float(tau)),
                 float(delay), float(tau))
                for delay in DELAYS_S for tau in TAUS_S),
               key=lambda item: item[0])[1:]


def fit_z_correction(records, response, kind, ridge):
    """Equal-weight runs, ridge shrinkage, no penalty on the intercept."""
    features = np.concatenate([
        z_features(r["time"], r["target"], kind) / np.sqrt(len(r["time"]))
        for r in records
    ])
    residual = np.concatenate([
        (r["actual"][:, 2] - rollout(r["time"], r["target"][:, 2],
                                      r["actual"][0, 2], *response)) /
        np.sqrt(len(r["time"])) for r in records
    ])
    penalty = np.eye(features.shape[1]) * ridge
    penalty[0, 0] = 0
    return np.linalg.solve(features.T @ features + penalty,
                           features.T @ residual)


def select_z_correction(records):
    """Choose correction only by leave-one-run-out error inside training set."""
    candidates = ((kind, ridge) for kind in ("constant", "linear_xyz", "quadratic_xyz")
                  for ridge in (.1, 1.0))
    folds = []
    for held_out in records:
        training = [r for r in records if r is not held_out]
        response = fit_z_response(training)
        base = rollout(held_out["time"], held_out["target"][:, 2],
                       held_out["actual"][0, 2], *response)
        folds.append((training, held_out, response, base))
    scored = []
    for kind, ridge in candidates:
        errors = []
        for training, held_out, response, base in folds:
            weights = fit_z_correction(training, response, kind, ridge)
            corrected = base + z_features(held_out["time"], held_out["target"], kind) @ weights
            errors.append(float(np.sqrt(np.mean((corrected - held_out["actual"][:, 2])**2))))
        scored.append({"kind": kind, "ridge": ridge,
                       "mean_held_out_z_rmse_mm": float(np.mean(errors)),
                       "per_run_z_rmse_mm": errors})
    return min(scored, key=lambda item: item["mean_held_out_z_rmse_mm"]), scored


def plot(t, leader, predicted, corrected, actual, destination, recording_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ("#b8a89a", "#2356a5", "#d15b3f")
    labels = ("Leader / sent target", "Predicted follower", "Recorded follower")
    for series, color, label in zip((leader, predicted, actual), colors, labels):
        axes[0, 0].plot(series[:, 0], series[:, 1], color=color,
                        linewidth=1.4, label=label)
    axes[0, 0].set(xlabel="Base-frame X (mm)", ylabel="Base-frame Y (mm)",
                   title="XY path")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=8)
    for j, axis in enumerate("XYZ"):
        ax = axes[(j + 1) // 2, (j + 1) % 2]
        for series, color, label in zip((leader, predicted, actual), colors, labels):
            ax.plot(t, series[:, j], color=color, linewidth=1.2,
                    label=label if axis == "Z" else None)
        if axis == "Z":
            ax.plot(t, corrected[:, 2], color="#34845b", linewidth=1.5,
                    label="Corrected Z prediction")
            ax.legend(fontsize=8)
        ax.set(xlabel="Time since recording start (s)",
               ylabel=f"Base-frame {axis} (mm)", title=f"{axis} over time")
    fig.suptitle(f"Held-out follower prediction from leader motion · {recording_name}\n"
                 "Z correction fitted and selected on six other runs; X/Y unchanged")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--held-out-name", default="arm_motion_20261010_211211_cartesian.csv")
    args = parser.parse_args()
    if args.held_out_name not in NAMES:
        parser.error("--held-out-name must be an archived paired Cartesian recording")
    raw_rows, t, leader, actual = read_pasted(args.input)
    archived = read(SOURCE / args.held_out_name)
    if len(t) != len(archived["time"]) or not np.allclose(t, archived["time"], atol=1e-6, rtol=0):
        raise ValueError("Pasted times do not match the selected archived run")
    if not np.allclose(leader, archived["target"], atol=1e-5, rtol=0):
        raise ValueError("Pasted leader does not match the archived sent target")
    training_names = [name for name in NAMES if name != args.held_out_name]
    parameters = fit([read(SOURCE / name) for name in training_names])
    predicted = np.column_stack([
        rollout(t, leader[:, i], actual[0, i], *parameters[a])
        for i, a in enumerate("xyz")
    ])
    selected, selection_scores = select_z_correction([read(SOURCE / name) for name in training_names])
    z_weights = fit_z_correction([read(SOURCE / name) for name in training_names],
                                  parameters["z"], selected["kind"], selected["ridge"])
    corrected = predicted.copy()
    corrected[:, 2] += z_features(t, leader, selected["kind"]) @ z_weights
    assert np.array_equal(corrected[:, :2], predicted[:, :2])
    error = np.linalg.norm(predicted - actual, axis=1)
    corrected_error = np.linalg.norm(corrected - actual, axis=1)
    baseline_error = np.linalg.norm(leader - actual, axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predicted_follower.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for raw, time, lead, pred, revised, observed, distance, revised_distance in zip(
                raw_rows, t, leader, predicted, corrected, actual, error, corrected_error):
            writer.writerow([raw["source_row"], f"{time:.6f}",
                             *[f"{v:.6f}" for v in lead],
                             *[f"{v:.6f}" for v in pred],
                             *[f"{v:.6f}" for v in observed], f"{distance:.6f}",
                             f"{revised[2]:.6f}", f"{revised_distance:.6f}"])
    plot(t, leader, predicted, corrected, actual,
         args.output_dir / "prediction_vs_recorded.png", args.held_out_name)
    summary = {
        "input": str(args.input), "held_out_recording": args.held_out_name,
        "training_recordings": training_names,
        "samples": len(t), "duration_s": float(t[-1] - t[0]),
        "initial_follower_xyz_mm": actual[0].tolist(),
        "leader_equals_sent_target": True,
        "parameters": {a: {"delay_s": parameters[a][0], "tau_s": parameters[a][1]}
                       for a in "xyz"},
        "z_correction": {"selected_by": "leave-one-run-out on six training runs",
                         "selected": selected, "candidates": selection_scores,
                         "weights": z_weights.tolist(),
                         "x_y_predictions_unchanged": True},
        "held_out_3d_rmse_mm": float(np.sqrt(np.mean(error**2))),
        "corrected_held_out_3d_rmse_mm": float(np.sqrt(np.mean(corrected_error**2))),
        "held_out_xy_rmse_mm": float(np.sqrt(np.mean(np.sum(
            (predicted[:, :2] - actual[:, :2])**2, axis=1)))),
        "held_out_3d_mae_mm": float(np.mean(error)),
        "corrected_held_out_3d_mae_mm": float(np.mean(corrected_error)),
        "held_out_3d_p95_error_mm": float(np.percentile(error, 95)),
        "corrected_held_out_3d_p95_error_mm": float(np.percentile(corrected_error, 95)),
        "held_out_3d_max_error_mm": float(np.max(error)),
        "direct_leader_3d_rmse_mm": float(np.sqrt(np.mean(baseline_error**2))),
        "direct_leader_xy_rmse_mm": float(np.sqrt(np.mean(np.sum(
            (leader[:, :2] - actual[:, :2])**2, axis=1)))),
        "per_axis_rmse_mm": {a: float(np.sqrt(np.mean((predicted[:, i] - actual[:, i])**2)))
                             for i, a in enumerate("xyz")},
        "corrected_z_rmse_mm": float(np.sqrt(np.mean((corrected[:, 2] - actual[:, 2])**2))),
        "limits": [
            "The initial follower XYZ comes from this run; later follower samples are withheld until scoring.",
            "XYZ positions come from forward kinematics, not direct pen-tip measurement.",
            "This model predicts motion only; it does not infer paper contact, pressure, or ink.",
            "The Z correction is empirical and needs validation on further unseen sessions.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
