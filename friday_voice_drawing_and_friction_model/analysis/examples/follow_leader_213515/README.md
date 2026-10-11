# Second held-out leader-to-follower check

The 659-sample `arm_motion_20261010_213515_cartesian.csv` run was excluded
from training. The delayed XYZ response and Z-only spatial correction were
fitted and selected using the six other paired recordings. Only the first
follower XYZ sample from this run initialized the prediction; all later
follower samples were reserved for scoring.

| Metric | Original predictor | Z correction |
| --- | ---: | ---: |
| XY RMSE | 15.22 mm | 15.22 mm |
| Z RMSE | 13.49 mm | 7.98 mm |
| 3D RMSE | 20.34 mm | 17.18 mm |

`predicted_follower.csv` contains both predictions, the recorded follower,
and pointwise errors. `prediction_vs_recorded.png` compares them. This is a
motion prediction test, not a pen-contact or real writing validation.

Reproduce from the repository root:

```bash
python friday_voice_drawing_and_friction_model/analysis/predict_follow_leader.py \
  --input friday_voice_drawing_and_friction_model/data/recordings/cartesian/arm_motion_20261010_213515_cartesian.csv \
  --held-out-name arm_motion_20261010_213515_cartesian.csv \
  --output-dir friday_voice_drawing_and_friction_model/analysis/examples/follow_leader_213515
```
