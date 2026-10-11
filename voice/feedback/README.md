# Friday drawing feedback loop

This is a new opt-in entry point. The existing `voice/ellab_to_voice.py` is unchanged.

## What it does

Speech → ElevenLabs agent → structured client tool → saved routine or bounded planner → robot controller → run status → spoken user rating.

A positive rating records +1 and selects that revision for future requests. A negative rating records −1 and enables a correction proposal. Candidates remain separate until tested and approved. This is persistent human-feedback adaptation, not RL training or a measured writing-accuracy score. No automatic exploration occurs.

The 9 baseline is the saved shortened-stem version (that shortening was not physically tested). The smiley baseline is the exact tested high-lift version. Both routines return to rest and retain torque. Existing controller dependencies and calibration are required on the AMD station.

## Setup

Run from a checkout on the robot station in its LeRobot environment (with a microphone/speaker available):

```sh
pip install 'elevenlabs[pyaudio]' numpy
export ELEVENLABS_API_KEY='your-key'
export ELEVENLABS_AGENT_ID='your-agent-id'
python voice/feedback/elevenlabs_bridge.py
```

Default is dry run: no physical commands and no rewards. To enable hardware intentionally, launch with `--physical`. Do not run the old voice controller simultaneously. This does not configure SSH control from a separate laptop.

In the ElevenLabs dashboard, paste `agent_prompt.txt` into your agent instructions. Configure five CLIENT tools with **Wait for response** enabled. Names and parameter names must match:

| Tool | Parameters |
|---|---|
| draw | content:string, request_id:string; optional revision_id:string |
| status | none |
| stop | none |
| feedback | run_id:string, accepted:boolean, feedback:string |
| propose | run_id:string, changes:object |

`changes` accepts only numeric `lift_delta_mm` (0–10), `stem_scale` (0.8–1), `mouth_depth_scale` (0.85–1.15), `eye_spacing_scale` (0.85–1.15). Shape-specific keys are checked. All are relative to the frozen baseline; repeat requests cannot accumulate unlimited offsets. Agent interpretation supplies these fields. Unsupported/ambiguous requests should be clarified, not guessed into motor commands.

The bridge sends a clearly labeled application-status message after completion to trigger the feedback question. This uses the SDK's text-message channel; it is never counted as user feedback. Completion only means the controller finished, not that ink quality was correct. Network delivery is retried while the session is alive.

## Example conversation

1. “Draw a smiley face.” → `draw`, new request ID.
2. Controller completes → “Did that look right?”
3. “No, lift higher between the eyes.” → `feedback(accepted=false)` then `propose(changes={"lift_delta_mm":5})`.
4. Agent explains candidate; “Test that.” → `draw` with returned revision ID and new request ID.
5. “Yes, keep it.” → `feedback(accepted=true)`; candidate becomes preferred.

## Model and execution boundaries

The planner adjusts existing paper waypoints and solves the project's calibrated FK/IK. It checks the 170 × 235 mm drawing rectangle, finite data, residual and joint speed before saving. The existing robot runner checks calibration, torque, serial-port ownership, and tracking. Geometry/contact remain approximate; these checks do not establish actual tip clearance or collision freedom.

Do not use coding-agent generated Python as an executable response to speech. Claude or Codex can implement an additional adapter returning the same `changes` object; the ElevenLabs agent already handles supported corrections. A separate Claude/Codex API connection is not configured here. There is no autonomous code rewrite, neural-network retraining, pressure learning, or camera scoring.

State, ratings, candidate folders, and controller logs live under ignored `runs/voice_feedback/`. Back up that folder to retain learned preferences. Only one bridge should own it. Restarted in-flight runs become `interrupted_unknown`; they are not automatically resumed or rated. Stop is requested via SIGINT and confirmed when the process exits. If the robot controller cannot communicate, software cannot guarantee physical stop.

## Offline verification

```sh
python -m unittest discover -s voice/feedback -p 'test_*.py'
```

Tests mock completed physical status in isolated temporary state; no robot or ElevenLabs request is made. Live ElevenLabs account configuration and end-to-end physical operation have not been tested.

## ADR: keep adaptation constrained

Status: implemented prototype. Decision: versioned routines plus bounded parameters and explicit user ratings. Alternative: unrestricted online policy training or coding-agent edits. The chosen approach preserves successful drawings, associates each reward with one run, prevents duplicate tool retries from duplicating motion, and permits rollback by choosing a previous revision. It limits adaptation to the supported parameters; broader changes require engineering work.

References: https://elevenlabs.io/docs/eleven-agents/customization/tools/client-tools and https://elevenlabs.io/docs/eleven-agents/customization/events/client-to-server-events
