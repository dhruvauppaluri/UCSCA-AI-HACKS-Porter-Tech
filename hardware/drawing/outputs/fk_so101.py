"""Minimal URDF forward kinematics for the SO-101 gripper frame."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


URDF = Path(__file__).with_name("so101_new_calib.urdf")
if not URDF.exists():
    URDF = Path(__file__).parent.parent / "work" / "so101_new_calib.urdf"
ROOT = ET.parse(URDF).getroot()
JOINTS_BY_CHILD = {joint.find("child").attrib["link"]: joint for joint in ROOT.findall("joint")}
TIP_OFFSET = np.array([-7.49, 3.33, 37.58]) / 1000
PAPER_NORMAL = np.array([0.06211592, -0.00188501, 0.99806716])


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def axis_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
    ])


def tip_position(joints_deg):
    path = []
    link = "gripper_frame_link"
    while link != "base_link":
        joint = JOINTS_BY_CHILD[link]
        path.append(joint)
        link = joint.find("parent").attrib["link"]
    transform = np.eye(4)
    for joint in reversed(path):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        step = np.eye(4)
        step[:3, :3] = rpy_matrix(rpy)
        step[:3, 3] = xyz
        transform = transform @ step
        if joint.attrib["type"] in ("revolute", "continuous"):
            name = joint.attrib["name"]
            axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            rotation = np.eye(4)
            rotation[:3, :3] = axis_matrix(axis, math.radians(joints_deg[name]))
            transform = transform @ rotation
    return transform[:3, 3] + transform[:3, :3] @ TIP_OFFSET


def relative_height_mm(joints_deg, clear_joints_deg):
    return float((tip_position(joints_deg) - tip_position(clear_joints_deg)) @ PAPER_NORMAL * 1000)
