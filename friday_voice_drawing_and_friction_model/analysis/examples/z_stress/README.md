# Held-out Z predictor stress test

[`stress_test.png`](stress_test.png) summarizes the results. The
[`stress_summary.json`](stress_summary.json) and
[`stress_trials.csv`](stress_trials.csv) contain the full results. The script
fits each test predictor on six paired runs, then tests on the seventh. It
repeats randomized conditions 30 times per run with seed `20261011`.

The paired leader and follower samples are also randomly **subsampled**
together (keeping 50% or 25% of timestamps). Other tests add independent
Gaussian noise to each leader XYZ sample, or to the single initial follower
Z measurement. The later follower Z samples remain withheld for scoring.

| Perturbation | Mean held-out full-run Z RMSE | Mean first-2s Z RMSE |
| --- | ---: | ---: |
| Original recordings | 5.93 mm | 1.42 mm |
| Leader XYZ noise, σ = 0.1 mm | 5.93 mm | 1.43 mm |
| Leader XYZ noise, σ = 2 mm | 6.04 mm | 2.32 mm |
| Leader XYZ noise, σ = 10 mm | 7.57 mm | 5.73 mm |
| Keep 50% of paired timestamps at random | 6.05 mm | 1.86 mm |
| Keep 25% of paired timestamps at random | 6.68 mm | 3.00 mm |
| Keep every eighth paired timestamp | 7.77 mm | 4.10 mm |

The averages hide short failures: the largest pointwise held-out Z error is
43.8 mm on the original recordings, and 126.7 mm in one random 25%-retention
trial. Sparse input can miss an abrupt command transition even when its
overall RMSE looks moderate. This predictor should not yet be treated as a
reliable contact-height controller.

On the nearly stationary `213515` start, the original cumulative-travel gate
mistook tiny input jitter for motion: σ = 0.1 mm leader noise increased the
first-2s Z RMSE to about 4.1 mm. The revised running-maximum displacement
gate gives about **0.10 mm** under that condition. This is a targeted fix for
the observed instability, not evidence of real-world accuracy under noisy
commands. At σ = 10 mm, the same early error rises to roughly 12 mm.

The stress test compares each perturbed *input* against the follower recorded
for the **original input**. No real follower response exists for the altered
leader command. Thus the noise tests measure sensitivity to measurement error,
not prediction accuracy for a new physical trajectory. The model form and
deadband were selected after inspecting these recordings; a fresh hardware
session is needed for independent validation. The source XYZ coordinates are
forward-kinematics estimates; this does not test pen contact or pressure.

Reproduce from the repository root:

```bash
python friday_voice_drawing_and_friction_model/analysis/stress_test_z.py \
  --output-dir friday_voice_drawing_and_friction_model/analysis/examples/z_stress
python friday_voice_drawing_and_friction_model/analysis/plot_z_stress.py
```
