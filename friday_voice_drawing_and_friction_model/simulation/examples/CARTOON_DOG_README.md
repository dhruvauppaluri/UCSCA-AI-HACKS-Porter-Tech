# Unseen cartoon dog check

`cartoon_dog.py` defines a fixed, front-facing dog in the measured 170 × 235 mm
inset paper frame. The dog points were not fitted to the simulation output.
The same SO-101 mesh, follower calibration, pen-tip/paper fit, actuator model,
speed, pen depth, and IK settings used for `HELLO WORLD` were used here.

The primary check uses **strict MuJoCo pen-paper contact only**. The earlier
5 mm near-contact visualization is retained in `cartoon_dog_mujoco_near5/` for
traceability but is excluded from the main score and comparison.

| Metric | Lag correction off | Lag correction on |
| --- | ---: | ---: |
| Timed pen-down XY RMSE | 2.90 mm | 4.87 mm |
| Intended stroke samples within 3 mm of collision ink | 90.5% | 94.5% |
| Intended stroke samples within 5 mm of collision ink | 100% | 100% |
| Pen-down samples without collision ink | 2 | 7 |
| Collision samples during planned pen-up motion | 29 | 31 |
| Those pen-up collisions over 3 mm from any intended stroke sample | 1 | 6 |

The coverage metric is sample-based nearest-neighbor distance on the paper;
it is not image-level stroke coverage or real ink. The path RMSE compares tip
position to the intended point at the same time. The two metrics can disagree:
an ink trace can resemble the shape while being early or displaced in time.

**Result:** The simulation draws a recognizable dog, but the empirical lag
correction does not improve this new task's timed tracking and causes more
off-stroke contact. It has not passed a held-out physical writing test. The
paper pose uses a replay-alignment hypothesis, and saved pivot/corner fitting
errors are about 9.8/13.2 mm RMS. Real-world accuracy is undetermined.

Reproduce both strict runs from the repository root:

```bash
python friday_voice_drawing_and_friction_model/simulation/simulate_text_mujoco.py \
  --drawing cartoon-dog --near-contact-mm 0 --lag-compensation off \
  --output-dir friday_voice_drawing_and_friction_model/simulation/examples/cartoon_dog_mujoco_uncompensated
python friday_voice_drawing_and_friction_model/simulation/simulate_text_mujoco.py \
  --drawing cartoon-dog --near-contact-mm 0 --lag-compensation on --video \
  --output-dir friday_voice_drawing_and_friction_model/simulation/examples/cartoon_dog_mujoco_strict
python friday_voice_drawing_and_friction_model/simulation/compare_dog_runs.py
```

Interact with both runs on macOS:

```bash
mjpython friday_voice_drawing_and_friction_model/simulation/view_text_interactive.py \
  --run-dir friday_voice_drawing_and_friction_model/simulation/examples/cartoon_dog_mujoco_strict \
  --compare-run-dir friday_voice_drawing_and_friction_model/simulation/examples/cartoon_dog_mujoco_uncompensated
```

P pauses, left/right steps, R restarts, B switches the two runs, I toggles ink,
T toggles the intended path, and C resets the camera.
