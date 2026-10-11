"""Replay a recorded SO-101 follower in MuJoCo and trace pen-paper contacts.

The source CSV stores calibrated LeRobot joint positions, not measured ink or
contact. This script fits a provisional pen tip and paper frame from the saved
pivot/corner captures, then runs either observed-pose replay or position-actuator
simulation. Its ink trace is conditional on those uncertain calibrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCE = ROOT / "data/recordings/raw/arm_motion_20261010_214924.csv"
ARM_XML = REPO / "so101_sim/model/so101_new_calib.xml"
CALIBRATION = ROOT / "data/calibration"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
BIAS_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def load_recording(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("recording needs at least two rows")
    t = np.array([float(row["time_s"]) for row in rows])
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("recording times must be finite and increasing")
    servo = json.loads((CALIBRATION / "follower_my_awesome_follower_arm.json").read_text())
    if any(servo[name]["id"] != i + 1 for i, name in enumerate((*JOINTS, "gripper"))):
        raise ValueError("follower calibration joint IDs do not match the recording")
    observed = np.array([[float(row[f"follower_{name}.pos"]) for name in JOINTS] for row in rows])
    commanded = np.array([[float(row[f"target_{name}.pos"]) for name in JOINTS] for row in rows])
    grip_obs = np.array([float(row["follower_gripper.pos"]) for row in rows])
    grip_cmd = np.array([float(row["target_gripper.pos"]) for row in rows])
    if not all(np.all(np.isfinite(v)) for v in (observed, commanded, grip_obs, grip_cmd)):
        raise ValueError("recording contains non-finite joint positions")
    # These CSV columns have already passed through LeRobot's servo calibration.
    # The JSON's homing offsets and tick ranges must not be applied again.
    return t, observed, commanded, grip_obs, grip_cmd


def make_fk(model):
    data = mujoco.MjData(model)
    qids = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    site = model.site("gripperframe").id

    def pose(degrees):
        data.qpos[qids] = np.deg2rad(degrees)
        mujoco.mj_forward(model, data)
        return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()

    return pose


def fit_pen_tip(pose):
    pivot = json.loads((CALIBRATION / "pivot.json").read_text())
    samples = np.array([[item["joint_positions"][name + ".pos"] for name in JOINTS]
                        for item in pivot.values()], dtype=float)
    bias_columns = [JOINTS.index(name) for name in BIAS_JOINTS]
    # Unknowns: four small joint-zero biases (deg), tip vector in the MuJoCo
    # gripper-site frame (mm), and the stationary pivot location (mm).
    x = np.array([0, 0, 0, 0, 37.58, 3.33, 7.49, 266.04, -98.52, -24.18], dtype=float)
    steps = np.array([.01] * 4 + [.1] * 6)

    def residual(value):
        bias = np.zeros(len(JOINTS))
        bias[bias_columns] = value[:4]
        tips = []
        for q in samples:
            position, rotation = pose(q + bias)
            tips.append(position * 1000 + rotation @ value[4:7])
        errors = (np.array(tips) - value[7:10]).ravel()
        return np.r_[errors, value[:4] * .5, (np.linalg.norm(value[4:7]) - 38.1) * .2]

    for _ in range(100):
        r = residual(x)
        jac = np.column_stack([(residual(x + np.eye(10)[i] * h) -
                                residual(x - np.eye(10)[i] * h)) / (2 * h)
                               for i, h in enumerate(steps)])
        delta = np.linalg.solve(jac.T @ jac + .01 * np.eye(10), -jac.T @ r)
        accepted = False
        for scale in (1., .5, .25, .1):
            trial = x + delta * scale
            if np.linalg.norm(residual(trial)) < np.linalg.norm(r):
                x = trial
                accepted = True
                break
        if not accepted or np.linalg.norm(delta) < 1e-5:
            break
    bias = dict(zip(BIAS_JOINTS, x[:4].tolist()))
    distances = residual(x)[:len(samples) * 3].reshape(-1, 3)
    return bias, x[4:7] / 1000, float(np.sqrt(np.mean(np.sum(distances**2, axis=1))))


def joint_bias_vector(bias):
    return np.array([bias.get(name, 0.0) for name in JOINTS])


def tip_positions(pose, joint_degrees, bias, tip_site_m):
    bias_vector = joint_bias_vector(bias)
    return np.array([position + rotation @ tip_site_m
                     for q in joint_degrees
                     for position, rotation in [pose(q + bias_vector)]])


def fit_paper(pose, bias, tip_site_m, paper_config):
    corners = json.loads((CALIBRATION / "corners.json").read_text())
    width = paper_config["usable_drawing_rectangle_mm"]["width"] / 1000
    height = paper_config["usable_drawing_rectangle_mm"]["height"] / 1000
    labels = ("bottom_left", "bottom_right", "top_left", "top_right")
    ideal = np.array([[0, 0, 0], [width, 0, 0], [0, height, 0], [width, height, 0]])
    measured = np.array([tip_positions(
        pose, np.array([[corners[label]["joint_positions"][name + ".pos"] for name in JOINTS]]),
        bias, tip_site_m)[0] for label in labels])
    u, _, vt = np.linalg.svd((ideal - ideal.mean(axis=0)).T @ (measured - measured.mean(axis=0)))
    row_rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    paper_to_world = row_rotation.T
    origin = measured.mean(axis=0) - ideal.mean(axis=0) @ row_rotation
    error_mm = 1000 * np.sqrt(np.mean(np.sum((ideal @ row_rotation + origin - measured)**2, axis=1)))
    return origin, paper_to_world, float(error_mm)


def paper_coordinates(world_points, origin, paper_to_world):
    return (np.asarray(world_points) - origin) @ paper_to_world


def replay_align_paper(origin, paper_to_world, recorded_tips, times, config, contact_window):
    coords = paper_coordinates(recorded_tips, origin, paper_to_world)
    start, end = contact_window
    mask = (times >= start) & (times <= end)
    sample = coords[mask]
    if not len(sample):
        raise ValueError("contact window does not overlap this recording")
    width = config["usable_drawing_rectangle_mm"]["width"] / 1000
    height = config["usable_drawing_rectangle_mm"]["height"] / 1000
    shift = np.array([(sample[:, 0].min() + sample[:, 0].max()) / 2 - width / 2,
                      (sample[:, 1].min() + sample[:, 1].max()) / 2 - height / 2,
                      np.quantile(sample[:, 2], .90)])
    return origin + paper_to_world @ shift, shift


def build_drawing_model(paper_origin, paper_to_world, tip_site_m, config):
    tree = ET.parse(ARM_XML)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.0025")
    option.set("gravity", "0 0 -9.81")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="800", offheight="800")
    gripper = next(body for body in root.iter("body") if body.get("name") == "gripper")
    frame = next(site for site in gripper.findall("site") if site.get("name") == "gripperframe")
    tool = ET.SubElement(gripper, "body", name="pen_tool", pos=frame.get("pos"), quat=frame.get("quat"))
    tip_string = " ".join(f"{value:.8f}" for value in tip_site_m)
    ET.SubElement(tool, "geom", name="pen_shaft", type="capsule",
                  fromto=f"0 0 0 {tip_string}", size="0.0025", mass="0.003",
                  rgba="0.15 0.17 0.19 1", contype="0", conaffinity="0")
    ET.SubElement(tool, "geom", name="pen_tip_geom", type="sphere", pos=tip_string,
                  size="0.0015", mass="0.001", rgba="0.04 0.05 0.10 1",
                  friction="0.3 0.005 0.0001")
    ET.SubElement(tool, "site", name="pen_tip_site", pos=tip_string, size="0.001")

    width = config["paper_width_mm"] / 1000
    height = config["paper_height_mm"] / 1000
    inset_width = config["usable_drawing_rectangle_mm"]["width"] / 1000
    inset_height = config["usable_drawing_rectangle_mm"]["height"] / 1000
    # Centering the inset on the sheet is an explicit assumption: the actual
    # offset from the outer paper edges was never measured.
    center = paper_origin + paper_to_world @ np.array([inset_width / 2, inset_height / 2, 0.])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, paper_to_world.ravel())
    world = root.find("worldbody")
    paper = ET.SubElement(world, "body", name="paper", pos=" ".join(map(str, center)),
                          quat=" ".join(map(str, quat)))
    ET.SubElement(paper, "geom", name="paper_visual", type="box",
                  size=f"{width/2} {height/2} 0.001", pos="0 0 -0.001",
                  rgba="0.96 0.96 0.92 1", contype="0", conaffinity="0")
    # An infinite collision half-space stays in contact even if recorded joint
    # poses place the tip below a thin paper box. Ink is still clipped to the
    # finite sheet dimensions below; penetration depth is reported as uncertain.
    ET.SubElement(paper, "geom", name="paper_geom", type="plane",
                  size=f"{width/2} {height/2} 0.001", pos="0 0 0",
                  rgba="1 1 1 0", friction="0.3 0.005 0.0001")
    # The drawable inset is a visual guide only; these lines do not collide.
    for name, pos, size in (
        ("bottom", (0, -inset_height / 2, .0001), (inset_width / 2, .0004, .0001)),
        ("top", (0, inset_height / 2, .0001), (inset_width / 2, .0004, .0001)),
        ("left", (-inset_width / 2, 0, .0001), (.0004, inset_height / 2, .0001)),
        ("right", (inset_width / 2, 0, .0001), (.0004, inset_height / 2, .0001)),
    ):
        ET.SubElement(paper, "geom", name=f"inset_{name}", type="box",
                      pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                      rgba="0.55 0.56 0.57 1", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="0.15 0 0.8", directional="true", dir="0 0 -1")
    camera_pos = center + paper_to_world[:, 2] * .45
    ET.SubElement(world, "camera", name="paper_overhead", pos=" ".join(map(str, camera_pos)),
                  quat=" ".join(map(str, quat)), fovy="44")
    angle_pos = center + paper_to_world @ np.array([.10, -.18, .38])
    backward = angle_pos - center
    backward /= np.linalg.norm(backward)
    right = np.cross(paper_to_world[:, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    angle_rotation = np.column_stack([right, up, backward])
    angle_quat = np.empty(4)
    mujoco.mju_mat2Quat(angle_quat, angle_rotation.ravel())
    ET.SubElement(world, "camera", name="paper_angle", pos=" ".join(map(str, angle_pos)),
                  quat=" ".join(map(str, angle_quat)), fovy="52")
    fd, temporary = tempfile.mkstemp(suffix=".xml", prefix="drawing_", dir=ARM_XML.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(temporary)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return model


def percent_to_joint(percent, model):
    lo, hi = model.jnt_range[model.joint("gripper").id]
    return lo + np.clip(np.asarray(percent) / 100, 0, 1) * (hi - lo)


def joint_arrays(model, observed, commanded, grip_observed, grip_commanded, bias):
    obs = np.column_stack([np.deg2rad(observed + joint_bias_vector(bias)),
                           percent_to_joint(grip_observed, model)])
    cmd = np.column_stack([np.deg2rad(commanded + joint_bias_vector(bias)),
                           percent_to_joint(grip_commanded, model)])
    # The archived shoulder-lift samples reach about -104 degrees while the
    # bundled MJCF limit is -100. Widen only as far as this recorded run needs.
    widened = {}
    for i, name in enumerate((*JOINTS, "gripper")):
        joint = model.joint(name).id
        actuator = model.actuator(name).id
        low = min(model.jnt_range[joint, 0], float(min(obs[:, i].min(), cmd[:, i].min()) - math.radians(1)))
        high = max(model.jnt_range[joint, 1], float(max(obs[:, i].max(), cmd[:, i].max()) + math.radians(1)))
        if low < model.jnt_range[joint, 0] or high > model.jnt_range[joint, 1]:
            widened[name] = [float(np.rad2deg(low)), float(np.rad2deg(high))]
        model.jnt_range[joint] = [low, high]
        model.actuator_ctrlrange[actuator] = [low, high]
    return obs, cmd, widened


def run_simulation(model, times, observed, commands, paper_origin, paper_to_world,
                   config, mode, contact_window, capture_video=False, capture_stride=3):
    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).id for name in (*JOINTS, "gripper")]
    qids = [model.jnt_qposadr[j] for j in joint_ids]
    vids = [model.jnt_dofadr[j] for j in joint_ids]
    site_id = model.site("pen_tip_site").id
    pen_id = model.geom("pen_tip_geom").id
    paper_id = model.geom("paper_geom").id
    data.qpos[qids] = observed[0]
    data.ctrl[:] = commands[0]
    mujoco.mj_forward(model, data)
    width = config["usable_drawing_rectangle_mm"]["width"] / 1000
    height = config["usable_drawing_rectangle_mm"]["height"] / 1000
    full_width = config["paper_width_mm"] / 1000
    full_height = config["paper_height_mm"] / 1000
    x_margin = (full_width - width) / 2
    y_margin = (full_height - height) / 2
    rows = []
    snapshots = []
    for i, t in enumerate(times):
        if mode == "replay":
            data.qpos[qids] = observed[i]
            if i:
                data.qvel[vids] = (observed[i] - observed[i - 1]) / (t - times[i - 1])
            data.ctrl[:] = commands[i]
            data.time = float(t)
            mujoco.mj_forward(model, data)
        else:
            while data.time + model.opt.timestep / 2 < t:
                data.ctrl[:] = [np.interp(data.time, times, commands[:, j]) for j in range(6)]
                mujoco.mj_step(model, data)
        tip = data.site_xpos[site_id].copy()
        local = paper_coordinates(tip, paper_origin, paper_to_world)
        contact_point = None
        for contact in data.contact[:data.ncon]:
            if {contact.geom1, contact.geom2} == {pen_id, paper_id}:
                contact_point = paper_coordinates(contact.pos, paper_origin, paper_to_world)
                break
        touching = contact_point is not None
        point = contact_point if touching else local
        on_sheet = (-x_margin <= point[0] <= width + x_margin and
                    -y_margin <= point[1] <= height + y_margin)
        in_bounds = (0 <= point[0] <= width and 0 <= point[1] <= height)
        ink = bool(touching and on_sheet)
        rows.append({"time_s": float(t), "paper_x_m": float(local[0]),
                     "paper_y_m": float(local[1]), "paper_z_m": float(local[2]),
                     "pen_paper_contact": int(touching), "ink": int(ink),
                     "ink_x_m": float(point[0]) if ink else None,
                     "ink_y_m": float(point[1]) if ink else None,
                     "inside_writable_bounds": int(in_bounds),
                     "source_contact_window": int(contact_window[0] <= t <= contact_window[1]),
                     "joint_error_deg": float(np.linalg.norm(np.rad2deg(data.qpos[qids[:5]] - observed[i, :5])))} )
        if capture_video and (i % capture_stride == 0 or i == len(times) - 1):
            snapshots.append((data.qpos.copy(), data.qvel.copy(), i + 1))
    return rows, data, snapshots


def write_trace(rows, destination):
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_ink(rows, config, destination, mode):
    width = config["usable_drawing_rectangle_mm"]["width"]
    height = config["usable_drawing_rectangle_mm"]["height"]
    full_width = config["paper_width_mm"]
    full_height = config["paper_height_mm"]
    x_margin = (full_width - width) / 2
    y_margin = (full_height - height) / 2
    fig, (ax, height_ax) = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1.2]})
    ax.add_patch(plt.Rectangle((-x_margin, -y_margin), full_width, full_height,
                               facecolor="#fffef8", edgecolor="#555", linewidth=1.5))
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False,
                               linestyle="--", edgecolor="#888", linewidth=1.1))
    points = np.array([[row["ink_x_m"] * 1000, row["ink_y_m"] * 1000]
                       for row in rows if row["ink"]])
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], s=5, color="#13269a", alpha=.8)
    ax.set_xlim(-x_margin - 5, width + x_margin + 5)
    ax.set_ylim(-y_margin - 5, height + y_margin + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Paper x (mm)")
    ax.set_ylabel("Paper y (mm)")
    ax.set_title("Blue = simulated contact marks; dashed = writable inset")
    t = np.array([r["time_s"] for r in rows])
    z = np.array([r["paper_z_m"] * 1000 for r in rows])
    mark = np.array([bool(r["ink"]) for r in rows])
    height_ax.plot(t, z, color="#444", linewidth=1)
    height_ax.scatter(t[mark], z[mark], s=5, color="#13269a")
    height_ax.axhline(0, color="#a44", linestyle="--", label="paper surface")
    height_ax.set_xlabel("Time (s)")
    height_ax.set_ylabel("Pen-tip center above paper (mm)")
    height_ax.set_title("Contact comes from MuJoCo geometry")
    height_ax.grid(alpha=.25)
    height_ax.legend()
    fig.suptitle(f"SO-101 {mode} · arm_motion_20261010_214924.csv\n"
                 "220 × 275 mm sheet; 170 × 235 mm writable inset (centered assumption)")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def render_scene(model, data, rows, paper_origin, paper_to_world, destination):
    """Render the simulated arm, finite sheet, inset, and accumulated ink."""
    renderer = mujoco.Renderer(model, width=800, height=800)
    try:
        renderer.update_scene(data, camera="paper_angle")
        for row in rows:
            if not row["ink"] or renderer.scene.ngeom >= renderer.scene.maxgeom:
                continue
            position = paper_origin + paper_to_world @ np.array(
                [row["ink_x_m"], row["ink_y_m"], .0004])
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([.0013, 0., 0.]), position,
                                np.eye(3).ravel(), np.array([0., .02, .15, 1.], dtype=np.float32))
            renderer.scene.ngeom += 1
        plt.imsave(destination, renderer.render())
    finally:
        renderer.close()


def render_video(model, snapshots, rows, paper_origin, paper_to_world, destination):
    """Animate the MuJoCo state and reveal simulated contact marks over time."""
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(model, width=640, height=640)
    render_data = mujoco.MjData(model)
    writer = imageio.get_writer(str(destination), fps=10, codec="libx264", quality=7,
                                macro_block_size=16)
    try:
        for qpos, qvel, row_count in snapshots:
            render_data.qpos[:] = qpos
            render_data.qvel[:] = qvel
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera="paper_angle")
            for row in rows[:row_count]:
                if not row["ink"] or renderer.scene.ngeom >= renderer.scene.maxgeom:
                    continue
                position = paper_origin + paper_to_world @ np.array(
                    [row["ink_x_m"], row["ink_y_m"], .0004])
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([.0013, 0., 0.]), position,
                                    np.eye(3).ravel(), np.array([0., .02, .15, 1.], dtype=np.float32))
                renderer.scene.ngeom += 1
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=SOURCE)
    parser.add_argument("--mode", choices=("replay", "driven"), default="replay",
                        help="Replay measured follower joints or simulate target-driven actuators")
    parser.add_argument("--paper-placement", choices=("captured", "replay-aligned"),
                        default="replay-aligned")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true",
                        help="Render a 10 fps MP4 of the MuJoCo arm and accumulating ink")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((CALIBRATION / "paper_config.json").read_text())
    results = json.loads((ROOT / "analysis/results.json").read_text())
    contact_window = (results["contact_start_s"], results["contact_end_s"])
    times, observed_deg, commanded_deg, grip_obs, grip_cmd = load_recording(args.recording)
    base_model = mujoco.MjModel.from_xml_path(str(ARM_XML))
    pose = make_fk(base_model)
    bias, tip_site_m, pivot_rms_mm = fit_pen_tip(pose)
    captured_origin, axes, corner_rms_mm = fit_paper(pose, bias, tip_site_m, config)
    observed_tip = tip_positions(pose, observed_deg, bias, tip_site_m)
    if args.paper_placement == "replay-aligned":
        paper_origin, shift = replay_align_paper(captured_origin, axes, observed_tip, times,
                                                 config, contact_window)
    else:
        paper_origin, shift = captured_origin, np.zeros(3)
    model = build_drawing_model(paper_origin, axes, tip_site_m, config)
    observed, commands, widened = joint_arrays(model, observed_deg, commanded_deg,
                                                 grip_obs, grip_cmd, bias)
    rows, data, snapshots = run_simulation(model, times, observed, commands, paper_origin,
                                           axes, config, args.mode, contact_window,
                                           capture_video=args.video)
    write_trace(rows, args.output_dir / "trace.csv")
    plot_ink(rows, config, args.output_dir / "ink_on_paper.png", args.mode)
    render_scene(model, data, rows, paper_origin, axes, args.output_dir / "mujoco_scene.png")
    if args.video:
        render_video(model, snapshots, rows, paper_origin, axes,
                     args.output_dir / "mujoco_writing.mp4")
    count = sum(row["ink"] for row in rows)
    summary = {
        "source_recording": str(args.recording.relative_to(REPO) if args.recording.is_relative_to(REPO)
                                else args.recording), "mode": args.mode,
        "samples": len(rows), "duration_s": float(times[-1]),
        "paper_mm": [config["paper_width_mm"], config["paper_height_mm"]],
        "writable_inset_mm": [config["usable_drawing_rectangle_mm"]["width"],
                              config["usable_drawing_rectangle_mm"]["height"]],
        "assumed_centered_inset_margins_mm": [
            (config["paper_width_mm"] - config["usable_drawing_rectangle_mm"]["width"]) / 2,
            (config["paper_height_mm"] - config["usable_drawing_rectangle_mm"]["height"]) / 2],
        "pen_tip_site_offset_mm": (tip_site_m * 1000).tolist(),
        "joint_zero_bias_deg": bias, "pivot_fit_rms_mm": pivot_rms_mm,
        "corner_fit_rms_mm": corner_rms_mm,
        "captured_paper_origin_world_m": captured_origin.tolist(),
        "paper_origin_world_m": paper_origin.tolist(),
        "paper_to_world_rotation": axes.tolist(),
        "replay_alignment_shift_in_captured_paper_frame_mm": (shift * 1000).tolist(),
        "joint_limits_widened_to_recording_deg": widened,
        "pen_paper_contact_samples": sum(r["pen_paper_contact"] for r in rows),
        "ink_samples": count,
        "ink_outside_writable_samples": sum(r["ink"] and not r["inside_writable_bounds"] for r in rows),
        "joint_tracking_rmse_deg": float(np.sqrt(np.mean([r["joint_error_deg"]**2 for r in rows]))),
        "limitations": ["CSV joint values were already servo-calibrated; tick offsets were not reapplied.",
                        "Contact, ink, paper pose, and pen offset were not directly measured in this recording.",
                        "Replay-aligned placement uses the same recording to shift the paper; it is a hypothesis.",
                        "Friction=0.3 and centered inset margins are unvalidated scene assumptions."],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("mode", "samples", "duration_s",
                                            "pivot_fit_rms_mm", "corner_fit_rms_mm",
                                            "pen_paper_contact_samples", "ink_samples",
                                            "ink_outside_writable_samples", "joint_tracking_rmse_deg")}, indent=2))
    print(f"Wrote trace.csv, ink_on_paper.png, mujoco_scene.png, and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
