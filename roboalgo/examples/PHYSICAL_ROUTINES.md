# Saved physical drawing routines

| Routine | Entry point | Status |
|---|---|---|
| 9 | `9_physical/saved_9/draw_9.py` | Preserves the operator-approved rounder loop; stem shortened by about one second of playback. Shortened variant not retested. |
| Smiley face | `smiley_physical/smiley_face/smiley_face.py` | Exact saved version of the high-lift physical test; two dots and one smile curve. |

Run only from the configured robot station, in its LeRobot environment, with the session's calibration and paper/pen arrangement. These commands move hardware:

```bash
python roboalgo/examples/9_physical/saved_9/draw_9.py
python roboalgo/examples/smiley_physical/smiley_face/smiley_face.py
```

Each script uses its adjacent CSV, approaches the starting point, executes the path, returns to the saved resting pose, and retains follower torque. Expected complete durations are approximately 23.4 seconds for the saved 9 and 38.1 seconds for the smiley. Runtime tracking failures stop the routine instead of guaranteeing a return.

Station dependencies include `~/drawing_capture/corner_capture.py`, `~/usefulcommands/play_dataforarm.py`, the LeRobot installation, and the follower's stable serial device and calibration. This is a station-specific experimental runner, not a portable hardware installation tool.

For smiley evidence see [the execution log](smiley_physical/high_lift_smiley_physical_20261011T183444Z.jsonl) and [saved report](smiley_physical/smiley_face/report.json). Intermediate folders preserve failed attempts, inferred geometry, and earlier candidates. Historical `NOT_EXECUTED` filenames describe their creation-time status; consult run logs and current summaries before interpreting them.
