import contextlib
import io
import json
import math
from pathlib import Path

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    from check_corners import fk

base = Path(__file__).parent
corners = json.loads((base / "corners.json").read_text())
names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
tip_offset = np.array([-7.49, 3.33, 37.58]) / 1000
limits = np.array([[-108, 108], [-98, 98], [-93, 93], [-94, 94], [-155, 155]], float)


def angles_of(label):
    return np.array([corners[label]["joint_positions"][name + ".pos"] for name in names])


def tip(q):
    positions = {name + ".pos": value for name, value in zip(names, q)}
    positions["gripper.pos"] = 0
    T = fk(positions)
    return T[:3, 3] + T[:3, :3] @ tip_offset


Q = {name: angles_of(name) for name in corners}
P = {name: tip(Q[name]) for name in corners}
normal = np.cross(P["bottom_right"] - P["bottom_left"], P["top_left"] - P["bottom_left"])
normal /= np.linalg.norm(normal)
if normal[2] < 0:
    normal = -normal


def bilinear(values, x, y):
    u = x / 170
    v = y / 235
    return ((1-u)*(1-v)*values["bottom_left"] + u*(1-v)*values["bottom_right"]
            + (1-u)*v*values["top_left"] + u*v*values["top_right"])


def solve(target, seed):
    q = seed.copy()
    for iteration in range(200):
        error = target - tip(q)
        if np.linalg.norm(error) < 0.0005:
            return q, np.linalg.norm(error) * 1000, iteration
        J = np.column_stack([(tip(q + np.eye(5)[i] * .03) - tip(q - np.eye(5)[i] * .03)) / .06
                             for i in range(5)])
        dq = J.T @ np.linalg.solve(J @ J.T + (0.0008**2) * np.eye(3), error)
        max_change = np.max(np.abs(dq))
        if max_change > 2:
            dq *= 2 / max_change
        q = np.clip(q + dq, limits[:, 0], limits[:, 1])
    return q, np.linalg.norm(target - tip(q)) * 1000, 200


path = []
corners_xy = [(35, 67.5), (135, 67.5), (135, 167.5), (35, 167.5), (35, 67.5)]
for edge in range(4):
    ax, ay = corners_xy[edge]
    bx, by = corners_xy[edge + 1]
    for step in range(20):
        f = step / 20
        x, y = ax*(1-f)+bx*f, ay*(1-f)+by*f
        target = bilinear(P, x, y) + normal * .045
        seed = bilinear(Q, x, y)
        q, error, iterations = solve(target, seed)
        path.append({"x_mm": x, "y_mm": y, "joints_deg": dict(zip(names, q.tolist())),
                     "ik_error_mm": error, "iterations": iterations,
                     "predicted_tip_mm": (tip(q)*1000).tolist()})
path.append(path[0].copy())

steps = [max(abs(path[i+1]["joints_deg"][n] - path[i]["joints_deg"][n]) for n in names)
         for i in range(len(path)-1)]
print("normal", normal)
print("max_ik_error_mm", max(item["ik_error_mm"] for item in path))
print("max_joint_step_deg", max(steps), "at", int(np.argmax(steps)))
print("start_joints_deg", path[0]["joints_deg"])
print("joint_ranges_deg", {n: [min(item["joints_deg"][n] for item in path),
                              max(item["joints_deg"][n] for item in path)] for n in names})
(base / "hover_square_plan.json").write_text(json.dumps({"side_mm": 100, "height_mm": 45,
                                                        "waypoints": path}, indent=2) + "\n")
