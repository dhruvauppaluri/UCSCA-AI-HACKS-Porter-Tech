# RoboAlgo: drawing trajectories and follower prediction

The predictor component packages the drawing project's **leader/command XYZ →
follower Z** algorithm. It reads a timed Cartesian command path in the robot
base frame and the follower's first measured Z position. It predicts later
follower Z positions without reading later follower measurements. It does not
alter X/Y, generate letters, solve inverse kinematics, or measure pen contact.

## Current drawing routines

The package also contains calibrated IK preparation and saved physical routines for **9** and **smiley face**. See [physical routines](examples/PHYSICAL_ROUTINES.md) and [voice feedback](../voice/feedback/README.md). Those replay controllers are separate from the offline Z predictor described below.

## Contents

- `z_model.py`: delayed first-order response plus a spatial Z correction. A
  running maximum displacement from the starting command, with a 1 mm
  deadband, turns on the correction after real movement.
- `predict.py`: apply the saved model to a new timed path.
- `evaluate.py`: compare a predicted Z trace with a paired follower recording;
  `plot_evaluation.py` plots the trace.
- `train.py`: fit the model on seven independent paired recordings and report
  leave-one-run-out errors.
- `stress_test.py`: repeat random measurement-noise and paired-sample-removal
  checks on held-out recordings; `plot_stress.py` draws the results.
- `artifacts/`: fitted JSON model, held-out validation and stress results,
  example CSV, and charts.
- `examples/21/`: timed “21” paper targets, simulated joint targets, and
  RoboAlgo predictions with the observed limitations for a physical test.

The seven source recordings remain in
`friday_voice_drawing_and_friction_model/data/recordings/cartesian/` in this
repository. Python needs `numpy`; plotting also needs `matplotlib`.

## Test with the physical robot

This is a **teleoperated recording and offline prediction** test. The model
does not command the follower. Use the same rigid pen mount, follower
calibration, and paper placement if you want to compare a new run with the
archived ones. The recorded Cartesian coordinates use the fitted pen-tip
offset and URDF `base_link` frame, not paper-plane height or pressure.

1. Pull this repository onto the robot computer. Confirm the SO-101 leader
   and follower calibrations are for `my_awesome_leader_arm` and
   `my_awesome_follower_arm`. The archived recorder uses stable USB IDs; see
   `hardware/drawing/robot-host/drawing_capture/corner_capture.py` and adjust
   them only if the device IDs actually changed. Keep the arm supported and
   the pen clear of the sheet at startup. The earlier hardware session had a
   shoulder-holding fault; stop if it sags or cannot hold position.
2. Make sure no other teleoperation process owns either serial port. From the
   repository root, start the recording loop with the robot's LeRobot Python:

   ```bash
   /home/amd-genai2/miniforge3/envs/lerobot/bin/python \
     hardware/drawing/robot-host/drawing_capture/standard_teleop_record.py
   ```

   Wait for `TELEOP READY; recording ...`. Move the leader slowly, do the
   intended test, lift the pen, then stop with Ctrl+C. The command prints the
   exact JSONL path under
   `hardware/drawing/robot-host/drawing_capture/data/`. Each row records a
   sent target and measured follower joints. The first five seconds blend
   from the measured startup pose, so keep that interval clear and include it
   when interpreting the trace.
3. On that computer or another clone with Python and NumPy, copy the new log
   into the converter's input directory. Set `RUN_ID` to the new file's stem
   shown by the recorder (for example `teleop_demo_20261011T035258Z`):

   ```bash
   RUN_ID=teleop_demo_YYYYMMDDTHHMMSSZ
   cp "hardware/drawing/robot-host/drawing_capture/data/$RUN_ID.jsonl" \
      hardware/drawing/remote-data/
   python hardware/drawing/outputs/convert_cartesian.py
   python roboalgo/evaluate.py \
     --input "hardware/drawing/cartesian/${RUN_ID}_cartesian.csv" \
     --output-dir "roboalgo/runs/$RUN_ID"
   python roboalgo/plot_evaluation.py --run-dir "roboalgo/runs/$RUN_ID"
   ```

   The converter also refreshes its other archived Cartesian exports and
   manifest. If recording happened outside the repo clone, copy the exact
   JSONL file from the robot computer first. Do not substitute a leader-only
   file: evaluation needs both sent targets and follower observations.
4. Read `roboalgo/runs/$RUN_ID/metrics.json`, `comparison.csv`, and
   `z_comparison.png`.
   `training_recording` should be `false` for a genuinely new run. Compare
   the predictor's Z RMSE, 95th-percentile absolute error, and maximum error
   with `direct_target_z_vs_recorded`. Inspect the plot for startup, lifts,
   and sudden moves. The first follower sample initializes the
   predictor; all later follower samples are used only for evaluation.

If you only want an open-loop prediction from a timed command CSV, use
`predict.py` below. It returns estimated Z but has no recorded follower trace
to score. The repository's earlier X/Y/Z response fit is in
`friday_voice_drawing_and_friction_model/analysis/all_cartesian_model.json`;
this package currently applies the newer correction only to Z.

From the repository root, run the saved model on a CSV containing `time_s`
and either `target_x_mm,target_y_mm,target_z_mm` or
`leader_model_x_mm,leader_model_y_mm,leader_model_z_mm`:

```bash
python roboalgo/predict.py \
  --model roboalgo/artifacts/z_predictor_all_seven.json \
  --input hardware/drawing/cartesian/arm_motion_20261010_213515_cartesian.csv \
  --initial-follower-z-mm 40.961762 \
  --output /tmp/predicted_follower_z.csv
```

The initial Z value in that example is the first follower reading from the
example recording. For a new run, use that run's first follower Z reading.

## Reproduce the archived analysis

To reproduce training and the stress test:

```bash
python roboalgo/train.py --output-dir roboalgo/artifacts
python roboalgo/stress_test.py --output-dir roboalgo/artifacts
python roboalgo/plot_stress.py
```

## What the measurements show

The mean of seven leave-one-run-out Z RMSE values is **5.93 mm**. With
independent Gaussian noise of σ = 0.1 mm on each leader XYZ sample, mean
held-out full-run Z RMSE remains **5.93 mm**; at σ = 10 mm it is **7.57 mm**.
Randomly keeping only 25% of paired timestamps gives **6.68 mm** mean RMSE,
but one pointwise error reaches **126.7 mm**. See
`artifacts/stress_summary.json` for all conditions and
`artifacts/stress_test.png` for the chart.

These are kinematic base-frame positions derived from robot readings. The
model and deadband were developed after examining the same recordings, so
held-out-run scores are diagnostics rather than independent new-session
validation. For perturbed commands, the recorded follower path is still the
one produced by the original command; those tests measure sensitivity to
input error, not the physical response to a newly altered command. The
predictor is not a verified pen-pressure or contact-height controller.
The local MuJoCo drawing experiment demonstrated paper mapping, pen-tip IK,
and joint targets in simulation, but that controller has not been packaged
into this folder or run against the physical arm. Autonomous pen-down testing
needs a separate hardware controller with live calibration, bounded joint
motion, and a verified way to hold/lift the pen before contact.
