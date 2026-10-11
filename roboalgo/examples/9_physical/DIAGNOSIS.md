# Failed physical paper test: 9

The user reported that the 2026-10-11 run produced a narrow peaked mark rather
than a 9. No repeat motion was performed for this diagnosis.

## Evidence

- The intended path is a smooth oval and tail at estimated paper Z = 0, with
  30 mm nominal approach/retraction heights. It was generated through archived
  forward/inverse kinematics. The learned XYZ/Z predictor did not modify these
  commands and no contact feedback corrected them.
- The log contains all 241 writing-phase samples over 7.9987 seconds. Median
  interval is 33.33 ms; maximum interval is 34.24 ms. Missing commands or a
  large scheduling stall do not explain the observed shape.
- Shoulder pan reported the same position for 3.767 seconds (path time
  4.235–8.002 s) while its commanded angle varied from -5.406 to -3.925 degrees.
  Across the writing phase, its observed span was 2.022 degrees compared with
  a commanded span of 3.373 degrees. This supports undertracking of small
  motions; static friction, backlash, controller deadband, and contact load
  cannot be separated using these position logs alone.
- In the archived FK/paper mapping, the intended 16 x 18 mm oval reconstructs
  from observed joints to spans of roughly 11.5 x 11.2 mm during the timed loop.
  Mean tail Y error is about +12.4 mm. These are model-derived estimates,
  not measurements of the user's photographed ink.
- The same reconstruction puts the repositioning path as low as -1.76 mm
  relative to the estimated sheet. The final 30 mm requested lift reconstructs
  to approximately 13 mm. Approach/retraction ink and intermittent loop contact
  are plausible, but the photo and servo log cannot identify precisely which
  movement produced each visible line.
- A later read-only motor check found torque disabled on all six motors and
  no latched status flags; PID registers read P=16, I=0, D=32. Torque had been
  enabled at run startup. There is no continuous torque-state log, so the later
  reading does not establish when it changed or that it caused this failure.

## Interpretation

CSV playback compatibility was established, but physical writing accuracy was
not. Tight numerical IK residuals only certify consistency with the assumed
geometry. Sending the entire command sequence does not verify the resulting
ink. The joint-error threshold used during execution was too coarse to judge
small-character quality. The uncertain paper/pen geometry also means modeled
clearance and contact must not be reported as observed physical clearance.

Before another drawing test, identify small-motion tracking and actual contact
height, then assess corrections using the observed follower response. This
report does not establish that faster or slower timing alone will fix the mark.

Sources: physical_9_20261011T175633Z.jsonl, physical_9_PLAYBACK.csv,
joint_diagnostics.json, the user's drawing image, and the archived paper frame.
