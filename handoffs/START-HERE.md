# Current handoff: robot drawing

Updated October 11, 2026. Start with the [project README](../README.md).

The active project is physical pen drawing and voice-guided feedback. Real SO-101 demonstrations, calibrations, generated trajectories, and execution logs are present.

## Continue work

1. Read [saved physical routines](../roboalgo/examples/PHYSICAL_ROUTINES.md). Preserve the saved 9 and smiley baselines.
2. Configure the five ElevenLabs client tools using [voice/feedback](../voice/feedback/README.md). The bridge is implemented but has only been tested offline.
3. Use the robot station's LeRobot environment and current calibration for physical execution. Host helper scripts remain dependencies; a fresh clone alone is not a complete station installation.
4. Keep feedback and candidates in `runs/voice_feedback/`; it is ignored by Git and needs separate backup to preserve learned preferences.

The operator accepted the rounder 9; the later shortened stem remains untested. The high-lift smiley completed a physical run and was selected to save. Joint playback completion and subjective approval are not quantitative ink-accuracy measurements.

The learned Z predictor remains an offline model. No trained drawing RL policy or independent Claude/Codex correction API is connected. The voice agent maps supported language into bounded planner parameters.
