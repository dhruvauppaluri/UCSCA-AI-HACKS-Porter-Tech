import sys, json
import numpy as np, pandas as pd

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else "../data/recordings/cartesian/arm_motion_20261010_214924_cartesian.csv"
d = pd.read_csv(CSV)
t = d.time_s.values
T = d[[f"target_{a}_mm" for a in "xyz"]].values
Fo = d[[f"follower_{a}_mm" for a in "xyz"]].values
dt = np.median(np.diff(t))

# ---- 1. contact detection: pen on paper = follower low and flat
z = Fo[:, 2]
contact = z < -15
# keep only the long contact block (smooth out blips)
idx = np.where(contact)[0]
c0, c1 = idx.min(), idx.max()
contact = np.zeros(len(t), bool); contact[c0:c1 + 1] = True
free = ~contact
print(f"contact window: {t[c0]:.2f}s -> {t[c1]:.2f}s ({contact.sum()} rows), free-air rows: {free.sum()}")

# ---- 2. servo response model: first-order lag  dp/dt = (target - p)/tau   (+ pure delay)
def simulate(tau, delay_steps, mask_axes=(0, 1, 2)):
    p = np.empty_like(Fo); p[0] = Fo[0]
    Td = np.vstack([np.repeat(T[:1], delay_steps, 0), T[:len(T) - delay_steps]]) if delay_steps else T
    a = dt / (tau + dt)
    for k in range(1, len(t)):
        p[k] = p[k - 1] + a * (Td[k] - p[k - 1])
    return p

best = None
for delay in range(0, 8):
    for tau in np.arange(0.01, 0.40, 0.005):
        p = simulate(tau, delay)
        err = np.mean(np.sum((p[free] - Fo[free]) ** 2, 1))
        if best is None or err < best[0]:
            best = (err, tau, delay)
_, TAU, DELAY = best
P = simulate(TAU, DELAY)            # where a friction-free arm WOULD be
print(f"servo model (fit on free-air rows): delay={DELAY*dt*1000:.0f} ms, time-constant tau={TAU*1000:.0f} ms, "
      f"free-air RMS fit error={np.sqrt(best[0]):.2f} mm")
raw_rms = lambda m: np.sqrt(np.mean(np.sum((Fo[m] - T[m]) ** 2, 1)))
res_rms = lambda m: np.sqrt(np.mean(np.sum((Fo[m] - P[m]) ** 2, 1)))
print(f"error vs target:        free {raw_rms(free):.1f} mm | contact {raw_rms(contact):.1f} mm")
print(f"error vs servo model:   free {res_rms(free):.1f} mm | contact {res_rms(contact):.1f} mm   <- leftover = contact effects")

R = Fo - P                           # residual displacement caused by contact
V = np.gradient(P, t, axis=0)        # pen velocity the servo is trying to produce
vxy = V[:, :2]; speed = np.linalg.norm(vxy, axis=1)

# ---- 3. PRESSURE: target pushed below the paper, follower held up by it
paper_z = np.median(Fo[contact, 2])
pen = np.clip(P[:, 2] - Fo[:, 2], None, None)        # model-predicted z minus actual z
press = -(R[:, 2])                                   # >0 when paper holds pen above where servo wants it
cmd_depth = np.clip(Fo[:, 2] - T[:, 2], 0, None)     # how far below actual pen the target is
print(f"\npaper height (follower z in contact): {paper_z:.1f} mm  (spread {np.std(Fo[contact,2]):.1f} mm)")
print(f"commanded depth below paper: mean {cmd_depth[contact].mean():.1f} mm, max {cmd_depth[contact].max():.1f} mm")
print(f"vertical residual (paper pushing back): mean {(-R[contact,2]).mean():.2f} mm vs free-air {(-R[free,2]).mean():.2f} mm")

# ---- 4. FRICTION: residual opposite to motion direction, in the paper plane
moving = contact & (speed > 10)          # mm/s
u = vxy[moving] / speed[moving, None]
drag = -np.sum(R[moving, :2] * u, 1)     # mm the pen lags BEHIND where friction-free arm would be
side = u[:,0]*R[moving,1] - u[:,1]*R[moving,0]      # sideways component (should average ~0)
A = np.c_[np.ones(moving.sum()), speed[moving]]
coef, *_ = np.linalg.lstsq(A, drag, rcond=None)
pred = A @ coef; r2 = 1 - np.sum((drag - pred) ** 2) / np.sum((drag - drag.mean()) ** 2)
fm = free & (speed > 10)
uf = vxy[fm] / speed[fm, None]; drag_free = -np.sum(R[fm, :2] * uf, 1)
print(f"\nmoving-in-contact rows: {moving.sum()}, speed {speed[moving].min():.0f}-{speed[moving].max():.0f} mm/s (median {np.median(speed[moving]):.0f})")
print(f"drag behind motion:  contact mean {drag.mean():.2f} mm  |  free-air mean {drag_free.mean():.2f} mm (control)")
print(f"sideways component:  mean {side.mean():.2f} mm (should be ~0)")
print(f"fit  drag = c0 + c1*speed :  c0 = {coef[0]:.2f} mm (Coulomb part),  c1 = {coef[1]*1000:.2f} mm per m/s (viscous part),  R^2 = {r2:.2f}")

depth_c = cmd_depth[moving]
mu_proxy = coef[0] / max(depth_c.mean(), 1e-6)
print(f"\nfriction ratio  (Coulomb drag / commanded depth, assuming similar stiffness in x,y,z): mu ~ {mu_proxy:.2f}")

out = dict(contact_start_s=float(t[c0]), contact_end_s=float(t[c1]), delay_ms=float(DELAY*dt*1000), tau_ms=float(TAU*1000),
           rms_free_raw=float(raw_rms(free)), rms_contact_raw=float(raw_rms(contact)),
           rms_free_model=float(res_rms(free)), rms_contact_model=float(res_rms(contact)),
           paper_z_mm=float(paper_z), cmd_depth_mean_mm=float(cmd_depth[contact].mean()),
           drag_contact_mm=float(drag.mean()), drag_free_mm=float(drag_free.mean()),
           c0_mm=float(coef[0]), c1_mm_per_mps=float(coef[1]*1000), r2=float(r2), mu_proxy=float(mu_proxy))
json.dump(out, open("results.json", "w"), indent=1)
np.savez("arrays.npz", t=t, T=T, Fo=Fo, P=P, R=R, contact=contact, speed=speed, moving=moving,
         drag=drag, drag_free=drag_free, speed_free=speed[fm], cmd_depth=cmd_depth, coef=coef)

# ---- 5. Coulomb check: does drag grow with how hard the pen is pressed?
dep = cmd_depth[moving]
A2 = np.c_[np.ones_like(dep), dep]
c2, *_ = np.linalg.lstsq(A2, drag, rcond=None)
p2 = A2 @ c2; r2b = 1 - np.sum((drag - p2) ** 2) / np.sum((drag - drag.mean()) ** 2)
# robust version: binned medians
bins = [0, 0.01, 3, 6, 9, 20]
bm = [(dep >= lo) & (dep < hi) for lo, hi in zip(bins, bins[1:])]
xm = np.array([np.median(dep[m]) for m in bm if m.sum() > 10]); ym = np.array([np.median(drag[m]) for m in bm if m.sum() > 10])
c3 = np.polyfit(xm, ym, 1)
print(f"\nCoulomb fit  drag = d0 + mu_eff*depth :  d0 = {c2[0]:.2f} mm, mu_eff = {c2[1]:.2f}, R^2 = {r2b:.2f}")
print(f"robust (binned medians):  d0 = {c3[1]:.2f} mm, mu_eff = {c3[0]:.2f}")
out.update(coulomb_d0_mm=float(c2[0]), coulomb_mu_eff=float(c2[1]), coulomb_r2=float(r2b),
           robust_d0_mm=float(c3[1]), robust_mu_eff=float(c3[0]))
json.dump(out, open("results.json", "w"), indent=1)
np.savez("arrays.npz", t=t, T=T, Fo=Fo, P=P, R=R, contact=contact, speed=speed, moving=moving,
         drag=drag, drag_free=drag_free, speed_free=speed[fm], cmd_depth=cmd_depth, coef=coef, c2=c2, xm=xm, ym=ym)
