import contextlib
import io
import json
from pathlib import Path

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import check_corners

fk = check_corners.fk
base = Path(__file__).parent
pivot = json.loads((base / "pivot.json").read_text())
corners = json.loads((base / "corners.json").read_text())
names = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def pose(item, bias):
    positions = item["joint_positions"].copy()
    for name, b in zip(names, bias):
        positions[name + ".pos"] += b
    return fk(positions)


def tip(item, x):
    T = pose(item, x[:4])
    return 1000 * T[:3, 3] + T[:3, :3] @ x[4:7]


def residual(x, regularize=False):
    errors = np.concatenate([tip(item, x) - x[7:10] for item in pivot.values()])
    if regularize:
        return np.r_[errors, x[:4] * .5, (np.linalg.norm(x[4:7]) - 38.1) * .2]
    return errors


def solve(regularize=False):
    x = np.r_[np.zeros(4), [-7.49, 3.33, 37.58], [266.04, -98.52, -24.18]]
    steps = np.r_[np.repeat(.01, 4), np.repeat(.1, 6)]
    for _ in range(100):
        r = residual(x, regularize)
        J = np.column_stack([(residual(x + np.eye(10)[i]*h, regularize) -
                              residual(x - np.eye(10)[i]*h, regularize))/(2*h)
                             for i, h in enumerate(steps)])
        delta = np.linalg.lstsq(J.T @ J + .01*np.eye(10), -J.T @ r, rcond=None)[0]
        accepted = False
        for step in (1., .5, .25, .1):
            y = x + delta*step
            if np.linalg.norm(residual(y, regularize)) < np.linalg.norm(r):
                x = y
                accepted = True
                break
        if not accepted or np.linalg.norm(delta) < 1e-5:
            break
    print("regularize", regularize)
    print("joint_zero_bias_deg", dict(zip(names, np.round(x[:4], 2))))
    print("pen_tip_offset_mm", np.round(x[4:7], 2), "length_mm", round(np.linalg.norm(x[4:7]), 2))
    per_pivot = {name: round(np.linalg.norm(tip(item, x) - x[7:10]), 2) for name, item in pivot.items()}
    print("pivot_error_mm", per_pivot)
    print("pivot_rms_mm", round(np.sqrt(np.mean(np.square(list(per_pivot.values())))), 2))
    q = {name: tip(item, x) for name, item in corners.items()}
    for a, b in (("bottom_left", "bottom_right"), ("top_left", "top_right"),
                 ("bottom_left", "top_left"), ("bottom_right", "top_right")):
        print(a, b, "mm", round(np.linalg.norm(q[b] - q[a]), 2))
    closure = q["top_right"] - q["bottom_right"] - q["top_left"] + q["bottom_left"]
    print("closure_mm", round(np.linalg.norm(closure), 2))


solve(False)
solve(True)
