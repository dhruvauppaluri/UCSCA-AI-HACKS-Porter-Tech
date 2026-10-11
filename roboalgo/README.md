# RoboAlgo: follower Z motion predictor

This folder packages the drawing project's current **leader/command XYZ →
follower Z** algorithm. It reads a timed Cartesian command path in the robot
base frame and the follower's first measured Z position. It predicts later
follower Z positions without reading later follower measurements. It does not
alter X/Y, generate letters, solve inverse kinematics, or measure pen contact.

## Contents

- `z_model.py`: delayed first-order response plus a spatial Z correction. A
  running maximum displacement from the starting command, with a 1 mm
  deadband, turns on the correction after real movement.
- `predict.py`: apply the saved model to a new timed path.
- `train.py`: fit the model on seven independent paired recordings and report
  leave-one-run-out errors.
- `stress_test.py`: repeat random measurement-noise and paired-sample-removal
  checks on held-out recordings; `plot_stress.py` draws the results.
- `artifacts/`: fitted JSON model, held-out validation and stress results,
  example CSV, and charts.

The seven source recordings remain in
`friday_voice_drawing_and_friction_model/data/recordings/cartesian/` in this
repository. Python needs `numpy`; plotting also needs `matplotlib`.

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
