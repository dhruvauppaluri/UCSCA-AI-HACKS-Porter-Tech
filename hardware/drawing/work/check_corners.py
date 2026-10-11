import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

root = ET.parse(Path(__file__).with_name("so101_new_calib.urdf")).getroot()
joints = {}
for joint in root.findall("joint"):
    joints[joint.find("child").attrib["link"]] = joint


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
    ])


def fk(positions):
    path = []
    link = "gripper_frame_link"
    while link != "base_link":
        joint = joints[link]
        path.append(joint)
        link = joint.find("parent").attrib["link"]
    T = np.eye(4)
    for joint in reversed(path):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        A = np.eye(4)
        A[:3, :3] = rpy_matrix(rpy)
        A[:3, 3] = xyz
        T = T @ A
        if joint.attrib["type"] in ("revolute", "continuous"):
            name = joint.attrib["name"]
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            B = np.eye(4)
            B[:3, :3] = axis_matrix(axis, math.radians(positions[name + ".pos"]))
            T = T @ B
    return T


captured = json.loads(Path(__file__).with_name("corners.json").read_text())
poses = {name: fk(item["joint_positions"]) for name, item in captured.items()}
for name, pose in poses.items():
    print(name, "gripper_frame_mm", np.round(pose[:3, 3] * 1000, 1))

for a, b in (("bottom_left", "bottom_right"), ("bottom_left", "top_left"),
             ("bottom_right", "top_right"), ("top_left", "top_right")):
    print(a, b, "gripper_frame_distance_mm",
          round(np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3]) * 1000, 1))

# The tip is at p_i + R_i*t, where t is an unknown fixed vector in the
# gripper frame. Check rectangle closure, side lengths, and perpendicularity.
names = ["bottom_left", "bottom_right", "top_left", "top_right"]
def residual(t):
    q = {n: poses[n][:3, 3] + poses[n][:3, :3] @ t for n in names}
    u = q["bottom_right"] - q["bottom_left"]
    v = q["top_left"] - q["bottom_left"]
    closure = q["top_right"] - q["bottom_right"] - q["top_left"] + q["bottom_left"]
    return np.r_[closure, np.linalg.norm(u) - .170, np.linalg.norm(v) - .235,
                 np.dot(u, v) / .170 / .235,
                 (np.linalg.norm(t) - .0381) * .2]

def jacobian(t):
    h = 1e-6
    return np.column_stack([(residual(t + np.eye(3)[i]*h)-residual(t - np.eye(3)[i]*h))/(2*h)
                            for i in range(3)])

best = None
for seed in (np.zeros(3), *(.0381 * np.eye(3)), *(-.0381 * np.eye(3))):
    t = seed.copy()
    for _ in range(100):
        r = residual(t)
        J = jacobian(t)
        step = np.linalg.lstsq(J.T @ J + 1e-4*np.eye(3), -J.T @ r, rcond=None)[0]
        t += step
        if np.linalg.norm(step) < 1e-8:
            break
    score = np.linalg.norm(residual(t))
    if best is None or score < best[0]:
        best = score, t

score, t = best
print("best_tip_offset_gripper_frame_mm", np.round(t*1000, 1), "length_mm", round(np.linalg.norm(t)*1000, 1))
print("fit_residual_m", np.round(residual(t), 4))
q = {n: poses[n][:3, 3] + poses[n][:3, :3] @ t for n in names}
for a, b in (("bottom_left", "bottom_right"), ("bottom_left", "top_left"),
             ("bottom_right", "top_right"), ("top_left", "top_right")):
    print(a, b, "tip_distance_mm", round(np.linalg.norm(q[a] - q[b]) * 1000, 1))
print("closure_error_mm", round(np.linalg.norm(q["top_right"] - q["bottom_right"] - q["top_left"] + q["bottom_left"]) * 1000, 1))

A = poses["top_right"][:3, :3] - poses["bottom_right"][:3, :3] - poses["top_left"][:3, :3] + poses["bottom_left"][:3, :3]
b = poses["top_right"][:3, 3] - poses["bottom_right"][:3, 3] - poses["top_left"][:3, 3] + poses["bottom_left"][:3, 3]
t_close = np.linalg.lstsq(A, -b, rcond=None)[0]
print("gripper_frame_closure_error_mm", round(np.linalg.norm(b)*1000, 1))
print("offset_for_exact_parallelogram_mm", np.round(t_close*1000, 1), "length_mm", round(np.linalg.norm(t_close)*1000, 1))
rng = np.random.default_rng(0)
candidates = rng.normal(size=(50000, 3))
candidates /= np.linalg.norm(candidates, axis=1)[:, None]
candidates *= .0381
errors = np.linalg.norm(candidates @ A.T + b, axis=1)
best_i = np.argmin(errors)
print("best_closure_with_38mm_offset_mm", round(errors[best_i]*1000, 1), "offset", np.round(candidates[best_i]*1000, 1))
