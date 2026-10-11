# Voice drawing and exploratory pen model

For the current physical drawing routines and ElevenLabs feedback bridge, start at the [project README](../README.md). The print-only voice interface below is an earlier modeling prototype.

This folder combines a 2D stroke planner, a voice interface, the recorded arm
data, follower calibration, response fits, and a paper-coordinate trajectory
simulator. The fitting scripts read the recordings bundled in this folder.

`data/recordings/raw/` contains the nine joint-space CSVs;
`data/recordings/cartesian/` contains their derived Cartesian paths and the
circle-session conversions. `data/recordings/session/` contains the full
teleoperation JSONL and its overlapping three-circle snapshot. The original
single-run video remains in `data/recordings/`. Paper dimensions, captured
corners, pivot sample, and servo calibration are in `data/calibration/`.
`simulation/examples/` contains the plotted `HELLO 42` comparison and its
generated CSVs. These are outputs of the model, not measured ink.

From this folder, install the numerical dependencies with
`python -m pip install -r requirements.txt`. The optional voice interface also
needs `elevenlabs[pyaudio]` and its configured credentials.

## Current pipeline

1. `voice/drawing.py` turns letters, words, digits, and a smiley into sampled 2D strokes.
2. `voice/friday_voice.py` sends those strokes to a print-only stub. It does not
   perform inverse kinematics or move the physical robot.
3. `analysis/fit_all_recordings.py` fits separate joint responses from all paired
   recordings. `analysis/fit_cartesian.py` fits XYZ responses and one shared
   response for paper-coordinate simulation. Both test each run after fitting
   on the other runs. Leader-only CSVs are inventoried but cannot measure follower
   tracking. The three-circle snapshot is omitted because it duplicates part of
   the full teleoperation run.
4. `analysis/analyze.py` contains the supplied single-run exploratory
   pressure/drag analysis. `analysis/results.json` holds its fitted numbers.
5. `simulation/trajectory_proxy.py` applies the shared XYZ response to a planned
   stroke path in the measured 170 × 235 mm inset paper rectangle. It emits target,
   predicted follower, and optional empirical drag-adjusted ink coordinates.

Example using NumPy:

```bash
python simulation/trajectory_proxy.py 'HELLO 42' --output /tmp/hello_predicted.csv
python simulation/trajectory_proxy.py 'HELLO 42' --compensate --output /tmp/hello_lag_compensated.csv
python simulation/trajectory_proxy.py 'HELLO 42' --compensate --drag --compensate-drag --output /tmp/hello_drag_hypothesis.csv
python analysis/fit_all_recordings.py  # refit from bundled raw/session data
python analysis/fit_cartesian.py       # refit from bundled Cartesian data
```

The simulation explicitly labels pen-down segments; it does not infer contact
from the joint data. Its paper origin is the inset rectangle's lower-left mark.
The target Z coordinate is relative to a nominal paper plane at 0 mm. The
fitted delay and time constant use seven independent paired recordings. The
optional drag fit uses one recording; it is an exploratory displacement proxy
and is off by default.

The optional `--compensate` mode uses a bounded look-ahead plus velocity term to
counter the fitted delay and lag. `--compensate-drag` adds an experimental shift
against the one-run drag proxy. The exported CSV includes desired, commanded,
predicted follower, and predicted ink positions. These are paper-plane simulation
coordinates, not inverse-kinematics joint commands.

## Measured fit quality

The nine archived CSVs comprise six paired runs and three leader-only runs. The
full two-arm circle-session JSONL adds a seventh independent paired run; its
shorter three-circle snapshot is overlapping data and is not fitted twice. The
seven paired runs contain 10,804 samples in total.
The leader-only files are mostly static or small adjustments; their joint ranges
and angular-speed summaries are in `analysis/all_recordings_model.json`.

In leave-one-run-out checks, the joint response had a sample-weighted combined
arm-angle RMSE of **3.59°** versus **7.19°** when using the current target directly.
The isotropic XYZ response had **19.65 mm** versus **25.64 mm**. Across individual
held-out runs, XYZ error ranged from roughly **14 to 32 mm**. These errors are
too large to support claims of accurate physical lettering.

A `HELLO 42` simulation produces 1,358 samples. Under the *same fitted lag model*
used to design the command, bounded lag compensation reduces simulated pen-down
XY error from 2.08 mm to 0.22 mm. This is an internal model check, not held-out
evidence. Turning on the one-run drag proxy raises the simulated ink error; its
compensation remains exploratory. Use `simulation/plot_comparison.py` to inspect
the intended strokes and predicted ink paths side by side.

## Limits of the archived fit

The archived analysis does not track the bundled video. It labels contact when
estimated robot-base Z is below a fixed threshold, then treats one continuous
interval as contact. The converted XYZ values themselves depend on an unvalidated
pen-tip offset and arm model. The 10 mm pivot residual and 22 mm corner closure
reported in the drawing archive exceed the precision implied by the plotted fits.

The single-run friction analysis compares predicted and observed **position**. It does not measure
normal force or tangential force, so the pressure and friction coefficients in
its README are hypotheses, not identified physical constants. Its servo fit is
also evaluated on the recording used to choose the parameters. The multi-run
response fit has held-out-run errors, but its XYZ still inherits arm and pen
calibration error. The video needs
time alignment and pen/paper tracking, followed by validation on a separate
recording, before using the numbers as predictive physics parameters.

The trajectory proxy is useful for exploring how candidate paths respond to
assumed lag and drag. It is not a validated robot controller. The repository's
separate `so101_sim/` contains a MuJoCo arm model and a gripper-frame IK
controller; pen-tip IK for arbitrary text remains to be connected.

## MuJoCo replay of the recorded circle session

`simulation/mujoco_recording_replay.py` uses the archived follower recording
`arm_motion_20261010_214924.csv`, follower calibration, captured pen-pivot and
paper-corner poses, and the SO-101 MuJoCo arm. It builds a 220 × 275 mm sheet
with a 170 × 235 mm writable inset. The inset is assumed centered, giving
25 mm side margins and 20 mm top/bottom margins; the paper config does not
record its exact offset. The script draws blue marks where the simulated pen
collides with the simulated paper and lies within the full sheet.

From this folder, recreate the actuator-driven video and trace with:

```bash
python simulation/mujoco_recording_replay.py --mode driven --video \
  --output-dir simulation/examples/arm_motion_214924_driven
```

The output includes `mujoco_writing.mp4`, `mujoco_scene.png`,
`ink_on_paper.png`, `trace.csv`, and `summary.json`. `--mode replay` places the
arm at each measured follower pose instead of integrating its actuator model.
`--paper-placement captured` uses the saved corner fit without translation.
The default `replay-aligned` placement translates that fit using the contact
window inferred from this recording. The two placements give materially
different ink results, so neither establishes the real paper position.

To explore the same run in MuJoCo's native 3D viewer on macOS:

```bash
mjpython simulation/view_recording_interactive.py
```

The viewer loops the actuator-driven replay. Drag to orbit, scroll to zoom,
press `P` to pause/resume, use the left/right arrows to step one recorded
sample while paused, `R` to restart, `I` to show/hide simulated ink, and `C`
to reset the camera. Close the window to stop. Use `--mode replay` to inspect
the measured follower poses instead of simulated actuator responses.

For an exploratory **5 mm near-contact allowance** (the middle of a requested
4–6 mm range), run:

```bash
python simulation/mujoco_recording_replay.py --mode driven --near-contact-mm 5 --video \
  --output-dir simulation/examples/arm_motion_214924_driven_near5
mjpython simulation/view_recording_interactive.py --near-contact-mm 5
```

This adds faint cyan marks when the simulated pen surface is within 5 mm of
the paper during the archived writing interval. Dark marks still mean an
actual MuJoCo collision. The allowance changes **only the ink interpretation**;
it does not change arm motion, collision forces, or measured pressure. Use `N`
in the viewer to hide/show faint marks. In this run, the allowance adds 75
inferred mark samples to 342 collision mark samples. It fills some gaps near
the tops of the right-hand strokes, but also adds isolated marks. In the
partial video comparison, predicted-to-observed mean distance increases from
6.2 to 6.7 mm, while observed-to-predicted mean distance decreases from 7.7
to 7.3 mm. So it covers more visible ink at the cost of additional predicted
marks; this is not a validated pressure model.

The video is a **conditional simulation**, not a reconstruction of measured ink.
The CSV contains joint positions and targets, but no pen contact, paper pose,
force, or ink observations. The pen-pivot fit has about 9.8 mm RMS residual and
the corner fit about 13.2 mm RMS residual. Friction is provisionally 0.3.
See `summary.json` for the exact alignment shift, contact counts, and limits.

### Check against the archived camera video

The archived 25.47 s video has the same run name and content as the newly
provided `IMG_4218.mov`. It already shows ink at its beginning, so a final
photo of the sheet cannot be scored against this one run. The video does show
new right-hand strokes appearing while the follower moves. Reproduce an
exploratory comparison of those strokes with:

```bash
python analysis/compare_recording_video.py
```

The script registers the 24 s frame to the 4 s frame, rectifies the sheet
using manually chosen corners, extracts newly darkened pixels, and compares
their approximate centerline with MuJoCo contact points from 12–20 s. The
result is `simulation/examples/arm_motion_214924_driven/video_vs_mujoco.png`
and `video_comparison.json`. With these specific choices, predicted points
are 6.2 mm from newly visible ink centerline on average, and 78% are within
10 mm. This describes a **partial, same-run visual fit**, not a general
writing accuracy percentage. Camera blur, pen occlusion, the hand-picked
paper corners, and the replay-aligned paper pose all affect the estimate.

## MuJoCo trajectory handoff

Generate a timed task-space CSV for another phrase with:

```bash
python simulation/export_mujoco_targets.py '67 67' \
  --output simulation/examples/67_67_mujoco_tip_targets.csv
```

The saved example has `time_s`, `paper_x_m`, `paper_y_m`, `paper_z_m`, and
`pen_down` (0 or 1). The origin is the **inset rectangle's bottom-left mark**;
+x points toward bottom-right, +y toward top-left, and +z away from the paper.
At 30 Hz, each row is a desired pen-tip waypoint, including pen-up travel and
the four pen-down digit strokes. The default 10 mm lift and 1 mm nominal
penetration are planning assumptions, not measured contact settings.

MuJoCo does not execute this CSV by itself. A drawing controller must transform
each paper-frame waypoint to a model-world position using the paper pose,
solve pen-tip IK, and send bounded joint targets to the actuators. It should
use simulated pen-paper contacts to decide what actually writes. `pen_down`
records the planner's intent, not a measured contact or an actuator command.
This CSV contains desired targets without Cartesian lag compensation; if the
MuJoCo actuators are tuned to the measured response, applying the same lag
model a second time would double-count it.
