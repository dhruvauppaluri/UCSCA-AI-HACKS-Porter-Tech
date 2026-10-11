# Voice drawing and exploratory pen model

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

The simulation is useful for exploring how candidate paths respond to assumed
lag and drag. It is not a validated robot controller or a MuJoCo implementation.
The repository's separate `so101_sim/` contains a MuJoCo arm model and a
gripper-frame IK controller; pen-tip IK and paper contact are not connected yet.
