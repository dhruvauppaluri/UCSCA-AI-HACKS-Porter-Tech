"""Reproducible held-out robustness checks for the follower Z predictor.

Perturbing recorded commands has no matching physical follower observation.
Scores therefore measure sensitivity to *input measurement error*, against
the follower from the original, unperturbed run—not accuracy for new motion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from z_model import NAMES, SOURCE, fit_model, predict_z, read_recording


SEED = 20261011
TRIALS = 30
NOISE_MM = (0.1, 0.5, 2.0, 5.0, 10.0)
START_NOISE_MM = (0.5, 2.0, 5.0)
DOWNSAMPLE = (2, 4, 8)
RANDOM_KEEP_FRACTION = (0.5, 0.25)


def evaluate(r, model, target, start_z, indices=None):
    if indices is None:
        indices = np.arange(len(r["time"]))
    t = r["time"][indices]
    actual = r["actual"][indices, 2]
    predicted = predict_z(t, target, start_z, model)
    delta = predicted - actual
    early = t - t[0] < 2
    return {"rmse_mm": float(np.sqrt(np.mean(delta * delta))),
            "first_2s_rmse_mm": float(np.sqrt(np.mean(delta[early] ** 2))),
            "p95_abs_error_mm": float(np.percentile(abs(delta), 95)),
            "max_abs_error_mm": float(np.max(abs(delta))),
            "nonfinite_samples": int(np.sum(~np.isfinite(predicted)))}


def summarize(records):
    return {"n": len(records),
            "mean_rmse_mm": float(np.mean([r["rmse_mm"] for r in records])),
            "mean_first_2s_rmse_mm": float(np.mean([r["first_2s_rmse_mm"] for r in records])),
            "p95_trial_rmse_mm": float(np.percentile([r["rmse_mm"] for r in records], 95)),
            "worst_trial_rmse_mm": float(max(r["rmse_mm"] for r in records)),
            "worst_point_error_mm": float(max(r["max_abs_error_mm"] for r in records)),
            "nonfinite_samples": int(sum(r["nonfinite_samples"] for r in records))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [read_recording(SOURCE / name) for name in NAMES]
    rng = np.random.default_rng(SEED)
    samples = []
    for held_out in runs:
        model = fit_model([r for r in runs if r is not held_out])
        actual_initial = held_out["actual"][0, 2]
        base = evaluate(held_out, model, held_out["target"], actual_initial)
        samples.append({"recording": held_out["name"], "scenario": "baseline",
                        "trial": 0, **base})
        for sigma in NOISE_MM:
            for trial in range(TRIALS):
                noisy = held_out["target"] + rng.normal(
                    0, sigma, held_out["target"].shape)
                samples.append({"recording": held_out["name"],
                                "scenario": f"leader_iid_noise_{sigma:g}mm",
                                "trial": trial,
                                **evaluate(held_out, model, noisy, actual_initial)})
        for sigma in START_NOISE_MM:
            for trial in range(TRIALS):
                initial = actual_initial + rng.normal(0, sigma)
                samples.append({"recording": held_out["name"],
                                "scenario": f"initial_z_noise_{sigma:g}mm",
                                "trial": trial,
                                **evaluate(held_out, model, held_out["target"], initial)})
        for step in DOWNSAMPLE:
            indices = np.unique(np.r_[np.arange(0, len(held_out["time"]), step),
                                      len(held_out["time"]) - 1])
            samples.append({"recording": held_out["name"],
                            "scenario": f"downsample_{step}x", "trial": 0,
                            **evaluate(held_out, model, held_out["target"][indices],
                                       actual_initial, indices)})
        for keep_fraction in RANDOM_KEEP_FRACTION:
            for trial in range(TRIALS):
                keep = rng.random(len(held_out["time"])) < keep_fraction
                keep[[0, -1]] = True
                indices = np.flatnonzero(keep)
                samples.append({"recording": held_out["name"],
                                "scenario": f"random_keep_{keep_fraction:g}",
                                "trial": trial,
                                **evaluate(held_out, model,
                                           held_out["target"][indices],
                                           actual_initial, indices)})
    scenarios = sorted({s["scenario"] for s in samples})
    summary = {name: summarize([s for s in samples if s["scenario"] == name])
               for name in scenarios}
    baseline = summary["baseline"]["mean_rmse_mm"]
    for name in scenarios:
        summary[name]["mean_rmse_increase_mm"] = summary[name]["mean_rmse_mm"] - baseline
    result = {"seed": SEED, "trials_per_run_per_random_scenario": TRIALS,
              "validation": "leave one entire run out when fitting each predictor",
              "reference": "actual follower Z from original unperturbed run",
              "caveat": "Perturbed leader commands have no matching physical follower recording; these are input robustness checks, not accuracy tests for altered motion.",
              "scenario_definitions": {
                  "leader_iid_noise": "Independent Gaussian measurement error on each leader XYZ sample and axis; follower recording unchanged.",
                  "initial_z_noise": "Gaussian error added only to the first follower Z measurement; later follower samples are held out.",
                  "downsample": "Keep every Nth paired sample plus the final sample; predict and score only at retained timestamps.",
                  "random_keep": "Randomly keep a stated fraction of paired timestamps (always including endpoints); predict and score at retained times."
              }, "scenarios": summary}
    (args.output_dir / "stress_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "stress_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
