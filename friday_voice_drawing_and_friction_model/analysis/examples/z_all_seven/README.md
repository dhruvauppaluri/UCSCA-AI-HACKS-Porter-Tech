# Follower Z predictor trained on all paired runs

The original Z correction was multiplied by a **time** ramp. On the `213515`
recording, its learned negative offset therefore turned on while the leader
was almost stationary. At about 1 second, the recorded follower was at
40.96 mm, but that correction predicted about 27.13 mm. The plot
[`early_z_comparison.png`](early_z_comparison.png) shows this early error.

The replacement uses the first measured follower Z to initialize the run,
holds that initial leader/follower offset while the leader is stationary,
and activates the learned spatial residual with the **largest displacement
from the leader's starting position**, after a 1 mm deadband. This running
maximum avoids accumulating small stationary sensor jitter into false motion.
The underlying delayed response and spatial coefficients were fitted to the
paired recordings. X and Y are untouched by this model. All coordinates are
forward-kinematics estimates in the robot base frame, not direct pen-tip
height measurements.

There are seven independent paired leader/follower recordings in the training
list of [`z_predictor_all_seven.json`](z_predictor_all_seven.json). Leader-only
files have no follower Z labels; a duplicated prefix of a paired run is not
an additional independent run. The JSON is the final model fitted on all
seven paired runs.

Before fitting the final JSON, each recording was held out once while the
other six were used to fit parameters. The mean of the seven held-out **run**
Z RMSE values is **5.93 mm**. For `213515`, held-out Z RMSE is **5.57 mm** over
the full 21.9 seconds and **0.002 mm** over the first 2 seconds. The old
time-ramped correction's full-run Z RMSE on that recording was **7.98 mm**.
See [`validation.json`](validation.json) for every run and
[`held_out_213515_z.csv`](held_out_213515_z.csv) for the plotted held-out
trajectory. The model form was chosen after looking at these runs, so these
scores are a cross-run diagnostic, not an independent new-session test. The
motion gate was also revised after seeing a noise stress test on these runs.
See [`../z_stress/README.md`](../z_stress/README.md) for that test.

To refit and reproduce the held-out checks from the repository root:

```bash
python friday_voice_drawing_and_friction_model/analysis/train_z_all_recordings.py \
  --output-dir friday_voice_drawing_and_friction_model/analysis/examples/z_all_seven
python friday_voice_drawing_and_friction_model/analysis/plot_z_early_fix.py
```

To predict Z for a new leader trajectory, provide the first follower Z
reading. No later follower measurements are used:

```bash
python friday_voice_drawing_and_friction_model/analysis/predict_z_from_model.py \
  --model friday_voice_drawing_and_friction_model/analysis/examples/z_all_seven/z_predictor_all_seven.json \
  --input path/to/leader_cartesian.csv \
  --initial-follower-z-mm 40.96 \
  --output path/to/predicted_z.csv
```

The model predicts the follower's base-frame Z from the recorded kinematic
coordinates. It has not been validated as a pen-pressure or ink-contact model.
