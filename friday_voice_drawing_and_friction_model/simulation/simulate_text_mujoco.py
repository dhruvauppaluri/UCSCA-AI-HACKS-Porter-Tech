"""Plan new text for the calibrated SO-101 and predict its MuJoCo ink.

This is an offline, follower-only experiment. The archived calibration, paper
alignment, and actuator model are hypotheses; output is not a hardware command.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_recording_replay import (
    ARM_XML, CALIBRATION, JOINTS, ROOT, SOURCE, build_drawing_model,
    fit_paper, fit_pen_tip, joint_arrays, load_recording, make_fk,
    paper_coordinates, replay_align_paper, render_scene, render_video,
    tip_positions,
)
from cartoon_dog import dog_strokes
from trajectory_proxy import build_target, build_target_from_strokes, compensate_command


def make_scene():
    config = json.loads((CALIBRATION / "paper_config.json").read_text())
    times, observed_deg, commanded_deg, grip_obs, grip_cmd = load_recording(SOURCE)
    base = mujoco.MjModel.from_xml_path(str(ARM_XML))
    pose = make_fk(base)
    bias, tip_offset, pivot_rms = fit_pen_tip(pose)
    captured_origin, axes, corner_rms = fit_paper(pose, bias, tip_offset, config)
    tips = tip_positions(pose, observed_deg, bias, tip_offset)
    fit = json.loads((ROOT / "analysis/results.json").read_text())
    origin, shift = replay_align_paper(captured_origin, axes, tips, times, config,
                                       (fit["contact_start_s"], fit["contact_end_s"]))
    model = build_drawing_model(origin, axes, tip_offset, config)
    observed, _, widened = joint_arrays(model, observed_deg, commanded_deg,
                                         grip_obs, grip_cmd, bias)
    return model, config, origin, axes, observed, tips, shift, bias, tip_offset, pivot_rms, corner_rms, widened


def solve_path(model, origin, axes, targets_m, observed, observed_tips, tip_offset):
    """Sequential damped pen-tip IK; keep the archived pen posture as a prior."""
    qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
    dids = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
    site = model.site("pen_tip_site").id
    data = mujoco.MjData(model)
    target0 = origin + axes @ targets_m[0]
    seed_index = int(np.argmin(np.linalg.norm(observed_tips - target0, axis=1)))
    q = observed[seed_index, :5].copy()
    posture = q.copy()
    gripper = float(observed[seed_index, 5])
    lower = model.jnt_range[[model.joint(n).id for n in JOINTS], 0] + .015
    upper = model.jnt_range[[model.joint(n).id for n in JOINTS], 1] - .015
    commands = np.empty((len(targets_m), 6))
    residuals = np.empty(len(targets_m))
    angle_errors = np.empty(len(targets_m))
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    frame = model.site("gripperframe").id
    data.qpos[qids] = posture
    mujoco.mj_forward(model, data)
    reference_axis = data.site_xmat[frame].reshape(3, 3) @ tip_offset
    reference_axis /= np.linalg.norm(reference_axis)
    for i, local_target in enumerate(targets_m):
        world_target = origin + axes @ local_target
        for _ in range(30):
            data.qpos[qids] = q
            mujoco.mj_forward(model, data)
            error = world_target - data.site_xpos[site]
            if np.linalg.norm(error) < .0008:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, site)
            axis = data.site_xmat[frame].reshape(3, 3) @ tip_offset
            axis /= np.linalg.norm(axis)
            skew_axis = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
            orientation_weight_m = .04
            jac = np.vstack([jacp[:, dids],
                             orientation_weight_m * (-skew_axis @ jacr[:, dids])])
            task_error = np.r_[error, orientation_weight_m * (reference_axis - axis)]
            # Preserve the recorded pen shaft direction while solving XYZ.
            damping = .006
            dq = np.linalg.solve(jac.T @ jac + damping**2 * np.eye(5),
                                 jac.T @ task_error)
            dq += .004 * (posture - q)
            q = np.clip(q + np.clip(dq, -.08, .08), lower, upper)
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        residuals[i] = np.linalg.norm(world_target - data.site_xpos[site]) * 1000
        axis = data.site_xmat[frame].reshape(3, 3) @ tip_offset
        axis /= np.linalg.norm(axis)
        angle_errors[i] = np.rad2deg(np.arccos(np.clip(axis @ reference_axis, -1, 1)))
        commands[i] = np.r_[q, gripper]
    return commands, residuals, angle_errors, seed_index


def drive(model, times, commands, observed, desired_m, pen_down, origin, axes,
          config, near_contact_mm):
    data = mujoco.MjData(model)
    names = (*JOINTS, "gripper")
    qids = [model.jnt_qposadr[model.joint(n).id] for n in names]
    pen_id = model.geom("pen_tip_geom").id
    paper_id = model.geom("paper_geom").id
    radius = float(model.geom_size[pen_id, 0])
    site = model.site("pen_tip_site").id
    data.qpos[qids] = commands[0]
    data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    width = config["usable_drawing_rectangle_mm"]["width"] / 1000
    height = config["usable_drawing_rectangle_mm"]["height"] / 1000
    margin_x = (config["paper_width_mm"] / 1000 - width) / 2
    margin_y = (config["paper_height_mm"] / 1000 - height) / 2
    rows, snapshots = [], []
    for i, t in enumerate(times):
        while data.time + model.opt.timestep / 2 < t:
            data.ctrl[:] = [np.interp(data.time, times, commands[:, j]) for j in range(6)]
            mujoco.mj_step(model, data)
        tip = paper_coordinates(data.site_xpos[site], origin, axes)
        contact_point = None
        for contact in data.contact[:data.ncon]:
            if {contact.geom1, contact.geom2} == {pen_id, paper_id}:
                contact_point = paper_coordinates(contact.pos, origin, axes)
                break
        actual_contact = contact_point is not None
        point = contact_point if actual_contact else tip
        on_sheet = (-margin_x <= point[0] <= width + margin_x and
                    -margin_y <= point[1] <= height + margin_y)
        clearance_mm = max(0., float(tip[2] - radius)) * 1000
        near = bool(not actual_contact and pen_down[i] and on_sheet and
                    clearance_mm <= near_contact_mm)
        # Off-stroke collisions are stray marks; near-contact inference requires
        # the planned pen-down label to avoid inventing travel strokes.
        ink = bool(on_sheet and (actual_contact or near))
        rows.append({
            "time_s": float(t), "desired_x_mm": desired_m[i, 0] * 1000,
            "desired_y_mm": desired_m[i, 1] * 1000,
            "desired_z_mm": desired_m[i, 2] * 1000,
            "paper_x_m": float(tip[0]), "paper_y_m": float(tip[1]),
            "paper_z_m": float(tip[2]), "pen_down": int(pen_down[i]),
            "pen_paper_contact": int(actual_contact),
            "inferred_near_contact": int(near),
            "pen_surface_clearance_mm": clearance_mm,
            "ink": int(ink), "ink_x_m": float(point[0]) if ink else None,
            "ink_y_m": float(point[1]) if ink else None,
            "inside_writable_bounds": int(0 <= point[0] <= width and 0 <= point[1] <= height),
        })
        if i % 4 == 0 or i == len(times) - 1:
            snapshots.append((data.qpos.copy(), data.qvel.copy(), i + 1))
    return rows, data, snapshots


def plot_result(rows, config, path, title):
    fig, ax = plt.subplots(figsize=(9, 11))
    width = config["usable_drawing_rectangle_mm"]["width"]
    height = config["usable_drawing_rectangle_mm"]["height"]
    margin_x = (config["paper_width_mm"] - width) / 2
    margin_y = (config["paper_height_mm"] - height) / 2
    ax.add_patch(plt.Rectangle((-margin_x, -margin_y), config["paper_width_mm"],
                               config["paper_height_mm"], facecolor="#fffef8",
                               edgecolor="#555"))
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False,
                               linestyle="--", edgecolor="gray"))
    desired = np.array([(r["desired_x_mm"], r["desired_y_mm"]) for r in rows])
    down = np.array([r["pen_down"] for r in rows], dtype=bool)
    ax.plot(np.where(down, desired[:, 0], np.nan),
            np.where(down, desired[:, 1], np.nan), color="#a1a1a1",
            linewidth=2.2, label="Requested stroke")
    for near, color, label in ((0, "#102b95", "Simulated collision ink"),
                               (1, "#62a6e8", "Possible light ink (5 mm allowance)")):
        points = np.array([(r["ink_x_m"] * 1000, r["ink_y_m"] * 1000)
                           for r in rows if r["ink"] and r["inferred_near_contact"] == near])
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=3, c=color, label=label)
    ax.set(xlim=(-margin_x - 5, width + margin_x + 5),
           ylim=(-margin_y - 5, height + margin_y + 5),
           xlabel="Paper x (mm)", ylabel="Paper y (mm)",
           title=f"{title}: requested path vs simulated ink")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="HELLO WORLD")
    p.add_argument("--drawing", choices=("text", "cartoon-dog"), default="text")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--near-contact-mm", type=float, default=5.)
    p.add_argument("--lag-compensation", choices=("on", "off"), default="on")
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    if not 0 <= args.near_contact_mm <= 20:
        p.error("--near-contact-mm must be between 0 and 20")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, config, origin, axes, observed, observed_tips, shift, bias, tip_offset, pivot_rms, corner_rms, widened = make_scene()
    path_settings = dict(sample_period_s=1 / 15, draw_speed_mm_s=35,
                         travel_speed_mm_s=50, lift_mm=10, press_depth_mm=1)
    if args.drawing == "cartoon-dog":
        times, desired_mm, pen_down = build_target_from_strokes(
            dog_strokes(), **path_settings)
        title = "CARTOON DOG"
    else:
        times, desired_mm, pen_down = build_target(args.text, **path_settings)
        title = args.text
    fit = json.loads((ROOT / "analysis/all_cartesian_model.json").read_text())
    response = fit["isotropic_parameters"]
    compensated_mm = (compensate_command(
        times, desired_mm, delay_ms=response["delay_s"] * 1000,
        tau_ms=response["tau_s"] * 1000)
        if args.lag_compensation == "on" else desired_mm.copy())
    commands, residuals, angle_errors, seed_index = solve_path(
        model, origin, axes, compensated_mm / 1000, observed, observed_tips,
        tip_offset)
    rows, data, snapshots = drive(model, times, commands, observed,
                                  desired_mm / 1000, pen_down, origin, axes,
                                  config, args.near_contact_mm)
    np.savez_compressed(args.output_dir / "viewer_states.npz",
                        qpos=np.array([s[0] for s in snapshots]),
                        qvel=np.array([s[1] for s in snapshots]),
                        row_count=np.array([s[2] for s in snapshots], dtype=int))
    with (args.output_dir / "trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "joint_commands.csv").open("w", newline="") as handle:
        names = ["time_s", *[f"{n}_rad" for n in (*JOINTS, "gripper")], "ik_residual_mm"]
        writer = csv.writer(handle)
        writer.writerow(names)
        for t, q, err in zip(times, commands, residuals):
            writer.writerow([t, *q, err])
    plot_result(rows, config, args.output_dir / "writing_comparison.png", title)
    render_scene(model, data, rows[::2], origin, axes,
                 args.output_dir / "mujoco_scene.png")
    if args.video:
        video_snapshots = [(qpos, qvel, (count + 1) // 2)
                           for qpos, qvel, count in snapshots]
        render_video(model, video_snapshots, rows[::2], origin, axes,
                     args.output_dir / "mujoco_writing.mp4")
    marked = np.array([bool(r["ink"]) for r in rows])
    down = np.array([bool(r["pen_down"]) for r in rows])
    xy_error_mm = np.array([np.hypot(
        r["paper_x_m"] * 1000 - r["desired_x_mm"],
        r["paper_y_m"] * 1000 - r["desired_y_mm"]) for r in rows])
    desired_xy = np.array([(r["desired_x_mm"], r["desired_y_mm"])
                           for r in rows])[down]
    collision_xy = np.array([(r["ink_x_m"] * 1000, r["ink_y_m"] * 1000)
                             for r in rows if r["ink"] and r["pen_paper_contact"]])
    nearest_collision_mm = (np.min(np.linalg.norm(
        desired_xy[:, None, :] - collision_xy[None, :, :], axis=2), axis=1)
        if len(collision_xy) else np.full(len(desired_xy), np.inf))
    offstroke_xy = np.array([(r["ink_x_m"] * 1000, r["ink_y_m"] * 1000)
                             for r in rows if r["ink"] and not r["pen_down"]])
    offstroke_far = int(np.sum(np.min(np.linalg.norm(
        offstroke_xy[:, None, :] - desired_xy[None, :, :], axis=2), axis=1) > 3)) if len(offstroke_xy) else 0
    summary = {
        "drawing": args.drawing, "text": args.text if args.drawing == "text" else None,
        "lag_compensation": args.lag_compensation,
        "samples": len(rows), "duration_s": float(times[-1]),
        "source_recording_for_calibration": str(SOURCE.relative_to(ROOT.parent)),
        "paper_mm": [config["paper_width_mm"], config["paper_height_mm"]],
        "writable_inset_mm": [config["usable_drawing_rectangle_mm"]["width"],
                              config["usable_drawing_rectangle_mm"]["height"]],
        "pen_tip_site_offset_mm": (tip_offset * 1000).tolist(),
        "joint_zero_bias_deg": bias,
        "pivot_fit_rms_mm": pivot_rms, "corner_fit_rms_mm": corner_rms,
        "replay_alignment_shift_mm": (shift * 1000).tolist(),
        "paper_origin_world_m": origin.tolist(), "paper_to_world_rotation": axes.tolist(),
        "seed_recording_index": seed_index,
        "joint_limits_widened_to_recording_deg": widened,
        "fitted_delay_s": response["delay_s"], "fitted_tau_s": response["tau_s"],
        "ik_residual_rms_mm": float(np.sqrt(np.mean(residuals**2))),
        "ik_residual_max_mm": float(np.max(residuals)),
        "pen_shaft_angle_change_median_deg": float(np.median(angle_errors)),
        "pen_shaft_angle_change_max_deg": float(np.max(angle_errors)),
        "ik_over_3mm_samples": int(np.sum(residuals > 3)),
        "requested_pen_down_samples": int(np.sum(down)),
        "simulated_pen_down_xy_rmse_mm": float(np.sqrt(np.mean(xy_error_mm[down]**2))),
        "simulated_pen_down_xy_max_error_mm": float(np.max(xy_error_mm[down])),
        "collision_stroke_sample_coverage_within_3mm_percent": float(
            100 * np.mean(nearest_collision_mm <= 3)),
        "collision_stroke_sample_coverage_within_5mm_percent": float(
            100 * np.mean(nearest_collision_mm <= 5)),
        "simulated_collision_ink_samples": int(sum(r["pen_paper_contact"] and r["ink"] for r in rows)),
        "inferred_near_contact_ink_samples": int(sum(r["inferred_near_contact"] for r in rows)),
        "pen_down_without_predicted_ink_samples": int(np.sum(down & ~marked)),
        "off_stroke_collision_ink_samples": int(sum(r["ink"] and not r["pen_down"] for r in rows)),
        "off_stroke_collision_over_3mm_from_stroke_samples": offstroke_far,
        "ink_outside_writable_bounds_samples": int(sum(r["ink"] and not r["inside_writable_bounds"] for r in rows)),
        "near_contact_allowance_mm": args.near_contact_mm,
        "limits": ["Paper pose was aligned to a prior recording and inset centering is assumed.",
                   "Pen tip and corner fit errors remain large relative to letter strokes.",
                   "Near-contact ink is a visual hypothesis, not a force or ink measurement.",
                   "Lag compensation is fitted to seven teleoperation runs; this new drawing has no real validation."],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "samples", "duration_s", "ik_residual_rms_mm", "ik_residual_max_mm",
        "ik_over_3mm_samples", "pen_shaft_angle_change_max_deg",
        "requested_pen_down_samples", "simulated_pen_down_xy_rmse_mm",
        "collision_stroke_sample_coverage_within_3mm_percent",
        "collision_stroke_sample_coverage_within_5mm_percent",
        "simulated_collision_ink_samples", "inferred_near_contact_ink_samples",
        "pen_down_without_predicted_ink_samples", "off_stroke_collision_ink_samples",
        "off_stroke_collision_over_3mm_from_stroke_samples")}, indent=2))


if __name__ == "__main__":
    main()
