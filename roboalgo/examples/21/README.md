# “21” drawing trajectory for physical test preparation

These CSVs describe **21** in the captured paper's 170 × 235 mm inset. Paper
origin is the inset bottom-left mark; +x points toward bottom-right, +y
toward top-left, and +z away from the sheet. The 220 × 275 mm sheet and
inset dimensions come from `friday_voice_drawing_and_friction_model/data/calibration/paper_config.json`.
The writing occupies x = 8.6–40.1 mm and y = 201.5–228.5 mm near the inset's
top-left corner.

| File | Meaning |
| --- | --- |
| `paper_waypoints.csv` | 265 timed paper-frame XYZ targets and `pen_down` flags over 17.6 s. Pen-up height is 10 mm; the planned pen-down tip depth is −1 mm relative to the nominal sheet plane. |
| `predicted_follower_xyz.csv` | Target base-frame XYZ, X/Y response-fit predictions, new RoboAlgo Z predictions, and the same MuJoCo run's tip position for comparison. All coordinates are **millimeters**. The prediction was initialized from the simulated first pose, not a measured physical pose. |
| `mujoco_joint_targets_SIM_ONLY.csv` | MuJoCo IK position targets in **radians**. These are not validated LeRobot `.pos` actuator commands. |
| `candidate_arm_angles_FOR_REVIEW.csv` | Approximate inverse-calibration angles in **degrees** for inspecting the servo mapping. It omits gripper commands and is intentionally not in the robot replay script's input format. |
| `source_sim/` | The original trace, joint targets, and summary needed to regenerate the derived CSVs. |
| `simulated_ink_preview.png` | Top-down view of requested strokes and strict MuJoCo collision ink. |
| `manifest.json` | Counts, calibration residuals, and prediction diagnostics. |

Rebuild the derived CSVs from the saved simulator trace:

```bash
python roboalgo/export_sim_trajectory.py \
  --sim-dir roboalgo/examples/21/source_sim \
  --output-dir roboalgo/examples/21
```

## Review before touching the robot

The strict-contact MuJoCo run produced a recognizable “21,” with 0.80 mm
maximum IK residual and 1.88 mm timed pen-down XY RMSE **inside that
simulation**. It also produced six contact samples during planned pen-up
motion. Its paper pose relies on a replay alignment; pen-tip and corner fits
have approximately 9.8 and 13.2 mm RMS residuals.

The new Z predictor does **not** agree with this new MuJoCo path: its Z RMSE
against the simulated tip is 5.93 mm. During pen-down samples, the predictor's
median paper-frame tip height is −4.42 mm while MuJoCo's tip center is at
+1.42 mm. Converting the simulated arm targets back through the archived
degree-based forward-kinematics model also misses the intended base-frame
tip positions by **8.29 mm RMS**. These discrepancies make the candidate
joint files unsuitable for direct pen-down replay on the physical follower.

For a physical comparison now, keep control with the existing teleoperation
recorder described in `roboalgo/README.md`, draw “21” manually, and evaluate
the fresh joint log against RoboAlgo. `paper_waypoints.csv` can serve as the
intended path reference. A later autonomous test needs a measured starting
follower pose, validated servo-angle mapping, live pen-clearance check, and
joint-motion bounds in a hardware controller. No file in this directory is a
verified executable writing command.
