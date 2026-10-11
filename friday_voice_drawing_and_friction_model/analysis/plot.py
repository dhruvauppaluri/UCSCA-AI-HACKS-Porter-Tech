import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
a = np.load("arrays.npz"); r = json.load(open("results.json"))
t, T, Fo, P, contact = a["t"], a["T"], a["Fo"], a["P"], a["contact"]
fig = plt.figure(figsize=(13, 10)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
for i, ax_n in enumerate("xz"):
    k = "xyz".index(ax_n); ax = fig.add_subplot(gs[i, :])
    ax.axvspan(r["contact_start_s"], r["contact_end_s"], color="#f3e7c9", label="pen on paper")
    ax.plot(t, T[:, k], color="#1f77b4", lw=1.2, label="target (leader)")
    ax.plot(t, P[:, k], color="#999", lw=1.2, ls="--", label=f"friction-free servo model (delay {r['delay_ms']:.0f} ms, τ {r['tau_ms']:.0f} ms)")
    ax.plot(t, Fo[:, k], color="#d62728", lw=1.4, label="follower (actual pen)")
    ax.set_ylabel(f"{ax_n} (mm)"); ax.grid(alpha=.3)
    if i == 0: ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_xlabel("time (s)")
ax = fig.add_subplot(gs[2, 0]); dep = a["cmd_depth"][a["moving"]]; drag = a["drag"]
ax.scatter(dep, drag, s=8, alpha=.35, color="#d62728", label="moving, pen down")
ax.scatter(a["xm"], a["ym"], s=60, color="k", zorder=3, label="binned median")
xx = np.linspace(0, dep.max(), 50); ax.plot(xx, r["coulomb_d0_mm"] + r["coulomb_mu_eff"] * xx, color="k", lw=1.5,
        label=f"fit: drag = {r['coulomb_d0_mm']:.1f} + {r['coulomb_mu_eff']:.2f}·depth")
ax.axhline(np.median(a["drag_free"]), color="#1f77b4", ls=":", label="pen in air (control)")
ax.set_xlabel("how far target pushes below paper (mm)  ∝ pressure"); ax.set_ylabel("pen drag behind motion (mm)")
ax.set_title("Friction grows with pressure (Coulomb)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(-20, 50)
ax = fig.add_subplot(gs[2, 1]); sp = a["speed"][a["moving"]]
ax.scatter(sp, drag, s=8, alpha=.35, color="#d62728")
ax.scatter(a["speed_free"], a["drag_free"], s=8, alpha=.35, color="#1f77b4", label="pen in air")
ax.set_xlabel("pen speed (mm/s)"); ax.set_ylabel("pen drag behind motion (mm)")
ax.set_title("…but barely depends on speed (little viscous drag)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(-20, 50)
fig.tight_layout(); fig.savefig("../docs/friction_analysis.png", dpi=110)
