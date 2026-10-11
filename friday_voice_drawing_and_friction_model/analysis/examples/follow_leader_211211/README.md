# Held-out follower prediction from the pasted 211211 recording

`predicted_follower.csv` contains the model-predicted follower XYZ for every
leader sample, alongside the recorded follower XYZ for evaluation. All XYZ
values are millimeters in the arm/base frame derived by forward kinematics.

The existing delayed first-order Cartesian model was fitted on the six other
paired recordings. The 211211 recording was excluded from fitting. Its first
follower XYZ sample initializes the rollout; later follower samples are used
only to score and plot the prediction. In this file, leader XYZ equals sent
target XYZ exactly, so the input to the rollout is the leader path.

| Held-out metric | Original | Z-only correction |
| --- | ---: | ---: |
| 3D prediction RMSE | 31.91 mm | 18.68 mm |
| Z prediction RMSE | 27.41 mm | 9.05 mm |
| XY prediction RMSE | 16.34 mm | 16.34 mm |
| 95th percentile 3D point error | 58.27 mm | 37.71 mm |
| Direct leader-to-follower 3D RMSE | 37.27 mm | — |

The Z-only correction is a ridge-fitted quadratic function of leader XYZ,
added after the original delayed response. A short ramp preserves the known
initial follower height. Its form and ridge strength were selected by
leave-one-run-out validation *within the six training runs* before evaluating
the pasted 211211 run. X and Y predictions are bit-for-bit unchanged.

The correction substantially reduces held-out Z error, but 9.05 mm is still
too large to establish pen-height control for writing. It is an empirical
position-dependent offset, not an identified pressure or contact law. Neither
contact nor ink is represented by this response model.

Reproduce from the repository root:

```bash
python friday_voice_drawing_and_friction_model/analysis/predict_follow_leader.py \
  --input '/Users/dhruvauppaluri/.codex/attachments/3abfbb49-6d6f-4f48-a86e-804a43e7e798/Pasted text.txt' \
  --output-dir friday_voice_drawing_and_friction_model/analysis/examples/follow_leader_211211
```
