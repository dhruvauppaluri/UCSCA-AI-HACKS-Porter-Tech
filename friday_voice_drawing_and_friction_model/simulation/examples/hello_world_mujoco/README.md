# HELLO WORLD follower-only MuJoCo run

The full SO-101 MJCF, follower calibration, pivot-derived pen offset, captured
paper corners, and replay-aligned paper pose are used to plan and simulate this
new text. `simulate_text_mujoco.py` converts glyph strokes to paper-frame targets,
applies the seven-run lag compensation, solves pen-tip IK while preserving the
recorded pen shaft direction, and drives MuJoCo's position actuators.

The 220 × 275 mm sheet and 170 × 235 mm writable inset come from
`data/calibration/paper_config.json`. The inset is assumed centered because its
offset from the sheet edges was not measured. The replay-based paper shift is
also a hypothesis, not an independently measured transform.

Run from the repository root with a Python environment containing MuJoCo,
NumPy, Matplotlib, ImageIO, and GLFW:

```bash
python friday_voice_drawing_and_friction_model/simulation/simulate_text_mujoco.py \
  --output-dir friday_voice_drawing_and_friction_model/simulation/examples/hello_world_mujoco --video
```

On macOS, open the interactive native viewer:

```bash
mjpython friday_voice_drawing_and_friction_model/simulation/view_text_interactive.py \
  --run-dir friday_voice_drawing_and_friction_model/simulation/examples/hello_world_mujoco
```

Drag to orbit and scroll to zoom. P pauses, arrow keys step, R restarts, I toggles
predicted ink, N toggles possible light marks, T toggles intended strokes, and C
resets the camera. The viewer plays at four times the planned writing speed.

`writing_comparison.png` is the most readable top-down comparison. Blue is
MuJoCo collision ink; light blue uses the 5 mm near-contact allowance. The
allowance does not change forces or prove real ink. `trace.csv` records tip
positions and contact decisions, `joint_commands.csv` records the follower-only
actuator targets, and `summary.json` records calibration and fit limits.

This run does not establish real-world accuracy. Pivot and corner fits have
9.8 and 13.2 mm RMS errors, respectively. The pen shaft angle changes by up to
5.5° from the recorded seed pose. The 2.0 mm pen-down path RMSE is internal to
the same simulation and excludes paper registration error. There are also 28
collision samples during planned pen-up motion, so the simulated writing is not
perfectly clean.
