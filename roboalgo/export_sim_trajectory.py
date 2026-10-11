"""Export paper targets and RoboAlgo predictions from a MuJoCo drawing run.

The resulting joint CSV is a simulation artifact, not a hardware command file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from z_model import ROOT, predict_z, rollout

sys.path.insert(0, str(ROOT / "hardware" / "drawing" / "outputs"))
from fk_so101 import tip_position  # noqa: E402


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace = read_rows(args.sim_dir / "trace.csv")
    joints = read_rows(args.sim_dir / "joint_commands.csv")
    sim = json.loads((args.sim_dir / "summary.json").read_text())
    if len(trace) < 2 or len(trace) != len(joints):
        raise ValueError("Trace and joint command files must contain the same nonzero trajectory")
    time = np.array([float(r["time_s"]) for r in trace])
    joint_time = np.array([float(r["time_s"]) for r in joints])
    if np.any(np.diff(time) <= 0) or not np.allclose(time, joint_time, atol=1e-6):
        raise ValueError("Trace and joint command timestamps do not align")
    paper_target = np.array([[float(r[f"desired_{axis}_mm"]) for axis in "xyz"]
                             for r in trace])
    paper_simulated = np.array([[float(r[f"paper_{axis}_m"]) * 1000 for axis in "xyz"]
                                for r in trace])
    pen_down = np.array([int(r["pen_down"]) for r in trace], dtype=int)
    origin = np.asarray(sim["paper_origin_world_m"], dtype=float) * 1000
    rotation = np.asarray(sim["paper_to_world_rotation"], dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Invalid simulated paper transform")
    target_base = origin + paper_target @ rotation.T
    simulated_base = origin + paper_simulated @ rotation.T
    xy_model = json.loads((ROOT / "friday_voice_drawing_and_friction_model" /
                           "analysis" / "all_cartesian_model.json").read_text())
    z_model = json.loads((Path(__file__).resolve().parent / "artifacts" /
                          "z_predictor_all_seven.json").read_text())
    predicted = np.empty_like(target_base)
    for axis_index, axis_name in enumerate("xy"):
        response = xy_model["parameters"][axis_name]
        predicted[:, axis_index] = rollout(
            time, target_base[:, axis_index], simulated_base[0, axis_index],
            response["delay_s"], response["tau_s"])
    predicted[:, 2] = predict_z(time, target_base, simulated_base[0, 2], z_model)
    predicted_paper = (predicted - origin) @ rotation
    width, height = sim["writable_inset_mm"]
    if (np.any(paper_target[:, 0] < 0) or np.any(paper_target[:, 0] > width) or
            np.any(paper_target[:, 1] < 0) or np.any(paper_target[:, 1] > height)):
        raise ValueError("Planned path exceeds the inset rectangle")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Nonfinite prediction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "paper_waypoints.csv",
               ("time_s", "paper_x_mm", "paper_y_mm", "paper_z_mm", "pen_down"),
               ((t, *p, d) for t, p, d in zip(time, paper_target, pen_down)))
    write_rows(args.output_dir / "predicted_follower_xyz.csv",
               ("time_s", "target_base_x_mm", "target_base_y_mm", "target_base_z_mm",
                "predicted_follower_x_mm", "predicted_follower_y_mm",
                "predicted_follower_z_mm", "simulated_follower_x_mm",
                "simulated_follower_y_mm", "simulated_follower_z_mm",
                "predicted_paper_z_mm", "simulated_paper_z_mm", "pen_down"),
               ((t, *target, *estimate, *observed, paper_estimate[2], paper_observed[2], d)
                for t, target, estimate, observed, paper_estimate, paper_observed, d in zip(
                    time, target_base, predicted, simulated_base,
                    predicted_paper, paper_simulated, pen_down)))
    write_rows(args.output_dir / "mujoco_joint_targets_SIM_ONLY.csv",
               ("time_s", *[f"{name}_rad" for name in JOINTS],
                "pen_down", "ik_residual_mm"),
               ((t, *[float(row[f"{name}_rad"]) for name in JOINTS], d,
                 float(row["ik_residual_mm"]))
                for t, row, d in zip(time, joints, pen_down)))
    arm_names = JOINTS[:5]
    bias = sim["joint_zero_bias_deg"]
    candidate_degrees = np.array([
        [np.rad2deg(float(row[f"{name}_rad"])) - float(bias.get(name, 0.))
         for name in arm_names] for row in joints])
    archived_fk = np.array([
        tip_position(dict(zip(arm_names, degree_row))) * 1000
        for degree_row in candidate_degrees])
    fk_error_mm = np.linalg.norm(archived_fk - target_base, axis=1)
    write_rows(args.output_dir / "candidate_arm_angles_FOR_REVIEW.csv",
               ("time_s", *[f"candidate_{name}_deg" for name in arm_names],
                "pen_down", "archived_fk_vs_target_error_mm"),
               ((t, *degree_row, d, error_mm)
                for t, degree_row, d, error_mm in zip(
                    time, candidate_degrees, pen_down, fk_error_mm)))
    error = predicted - simulated_base
    report = {
        "text": sim["text"], "samples": len(time), "duration_s": float(time[-1] - time[0]),
        "paper_mm": sim["paper_mm"], "writable_inset_mm": sim["writable_inset_mm"],
        "pen_down_samples": int(np.sum(pen_down)),
        "initial_simulated_follower_base_xyz_mm": simulated_base[0].tolist(),
        "prediction_vs_same_simulation_rmse_mm": {
            axis: float(np.sqrt(np.mean(error[:, i] ** 2))) for i, axis in enumerate("xyz")},
        "predicted_pen_down_paper_z_median_mm": float(np.median(predicted_paper[pen_down.astype(bool), 2])),
        "simulated_pen_down_paper_z_median_mm": float(np.median(paper_simulated[pen_down.astype(bool), 2])),
        "simulated_ik_residual_max_mm": sim["ik_residual_max_mm"],
        "simulated_off_stroke_collision_samples": sim["off_stroke_collision_ink_samples"],
        "candidate_joint_archived_fk_vs_target_rms_mm": float(
            np.sqrt(np.mean(fk_error_mm ** 2))),
        "candidate_joint_archived_fk_vs_target_max_mm": float(np.max(fk_error_mm)),
        "pivot_fit_rms_mm": sim["pivot_fit_rms_mm"],
        "corner_fit_rms_mm": sim["corner_fit_rms_mm"],
        "physical_joint_command_validated": False,
        "status": "analysis_only; the Z predictor diverges from the MuJoCo pen height on this new path",
        "note": "Initial follower XYZ came from MuJoCo. For a physical test, use a measured start pose; these joint targets are MuJoCo radians, not validated LeRobot actuator commands.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
