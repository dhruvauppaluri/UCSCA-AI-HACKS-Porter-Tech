"""Plan a small circle from a user-confirmed pen-contact servo pose."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
from fk_so101 import tip_position

ROOT = Path(__file__).resolve().parent
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CONTACT = {"shoulder_pan": 31.12087912087912, "shoulder_lift": 22.32967032967033,
           "elbow_flex": -9.142857142857142, "wrist_flex": 86.41758241758242,
           "wrist_roll": 18.76923076923077}
RADIUS_MM = 12
LIFT_MM = 20


def tip(q):
    return tip_position(dict(zip(NAMES, q)))


def solve(target, seed):
    q = seed.copy()
    for iteration in range(200):
        error = target - tip(q)
        if np.linalg.norm(error) < .00035:
            return q, np.linalg.norm(error)*1000
        jac = np.column_stack([(tip(q+np.eye(5)[i]*.03)-tip(q-np.eye(5)[i]*.03))/.06
                               for i in range(5)])
        weights = np.diag([1, 1, 1, 1, .15])
        change = weights @ jac.T @ np.linalg.solve(jac @ weights @ jac.T + .0008**2*np.eye(3), error)
        change *= min(1, .8/max(abs(change)))
        q = q + change
        q[4] = np.clip(q[4], CONTACT["wrist_roll"]-2, CONTACT["wrist_roll"]+2)
        if np.any(abs(q) > [108, 98, 93, 94, 155]):
            raise RuntimeError("joint limit exceeded")
    raise RuntimeError("IK did not converge")


def main():
    corners = json.loads((ROOT / "corners.json").read_text())
    positions = {name: tip(np.array([corners[name]["joint_positions"][n+".pos"]
                                      for n in NAMES], float)) for name in corners}
    xvec = positions["bottom_right"]-positions["bottom_left"]
    yvec = positions["top_left"]-positions["bottom_left"]
    xunit = xvec/np.linalg.norm(xvec)
    yunit = yvec - xunit*np.dot(yvec,xunit)
    yunit /= np.linalg.norm(yunit)
    normal = np.cross(xunit,yunit)
    if normal[2] < 0: normal = -normal
    origin = positions["bottom_left"]
    q0 = np.array([CONTACT[n] for n in NAMES],float)
    center = tip(q0)
    a = np.column_stack((xvec,yvec))
    xy = np.linalg.lstsq(a,center-origin,rcond=None)[0]*[170,235]
    if not (RADIUS_MM+10 < xy[0] < 170-RADIUS_MM-10 and
            RADIUS_MM+10 < xy[1] < 235-RADIUS_MM-10):
        raise RuntimeError(f"contact pose near inset boundary: {xy}")
    waypoints=[]
    seed=q0
    # Raise vertically from the contact center before any lateral travel.
    for i in range(1,11):
        point=center+normal*(LIFT_MM/1000*i/10)
        seed,error=solve(point,seed)
        waypoints.append({"phase":"lift","joints_deg":dict(zip(NAMES,seed.tolist())),"ik_error_mm":error})
    # Move over the circle's first point while lifted.
    for i in range(1,13):
        point=center+normal*(LIFT_MM/1000)+xunit*(RADIUS_MM/1000*i/12)
        seed,error=solve(point,seed)
        waypoints.append({"phase":"move_to_edge","joints_deg":dict(zip(NAMES,seed.tolist())),"ik_error_mm":error})
    # Descend to the calibrated contact plane at the edge.
    for i in range(1,21):
        point=center+xunit*(RADIUS_MM/1000)+normal*(LIFT_MM/1000*(1-i/20))
        seed,error=solve(point,seed)
        waypoints.append({"phase":"descend","joints_deg":dict(zip(NAMES,seed.tolist())),"ik_error_mm":error})
    # 48 steps around the circle; keeps the center untouched.
    for i in range(1,49):
        angle=2*math.pi*i/48
        point=center+(xunit*math.cos(angle)+yunit*math.sin(angle))*RADIUS_MM/1000
        seed,error=solve(point,seed)
        waypoints.append({"phase":"draw","joints_deg":dict(zip(NAMES,seed.tolist())),"ik_error_mm":error})
    for i in range(1,11):
        point=center+xunit*(RADIUS_MM/1000)+normal*(LIFT_MM/1000*i/10)
        seed,error=solve(point,seed)
        waypoints.append({"phase":"retract","joints_deg":dict(zip(NAMES,seed.tolist())),"ik_error_mm":error})
    print('center_xy_mm',xy,'radius_mm',RADIUS_MM,'max_error_mm',max(w['ik_error_mm'] for w in waypoints))
    print('first_lift',waypoints[0]['joints_deg'])
    print('last',waypoints[-1]['joints_deg'])
    (ROOT/'circle_plan.json').write_text(json.dumps({'contact_joints_deg':CONTACT,'center_xy_mm':xy.tolist(),
        'radius_mm':RADIUS_MM,'lift_mm':LIFT_MM,'waypoints':waypoints},indent=2)+'\n')


if __name__=='__main__':main()
