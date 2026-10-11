# Friday — voice-controlled drawing arm (SO-101 + LeRobot + ElevenLabs)

Say **"Friday, draw 42"** or **"Friday, draw a smiley face"** and an SO-101 arm holding a pen writes it on paper,
typewriter style: left to right, wrapping to the next line at the page edge.

AI Hackathon #SFTechWeek, AMD Track (Physical AI / Robotics), Oct 10–11 2026.

## Repo layout

| Folder | What's in it | Who uses it |
|---|---|---|
| `voice/` | `friday_voice.py` (ElevenLabs voice agent → tool calls), `drawing.py` (digits + smiley as piecewise line/arc functions on a coordinate plane) | voice team |
| `analysis/` | `analyze.py` + `plot.py`: servo-lag model and pen friction / pressure fit from recorded data; `results.json` = fitted numbers | modeling |
| `data/recordings/` | 25 s teleop recording of the follower drawing: Cartesian pen positions (target / follower / leader, mm, 30 Hz) + the video | modeling, MuJoCo |
| `data/calibration/` | LeRobot calibration files for the leader and follower arms | MuJoCo, arm |
| `docs/` | figures | everyone |

## Run the voice control

```bash
pip install -r requirements.txt          # Mac: brew install portaudio first
export ELEVENLABS_API_KEY="..."          # never commit this
export ELEVENLABS_AGENT_ID="..."
cd voice && python3 friday_voice.py
```
`ARM_MODE = "print"` simulates the drawing (writes `preview.png`). The arm team fills in `send_to_arm()`.

Agent tools (ElevenLabs client tools): `draw(content: string)` with digits or `"smiley"`, `new_page()`, `stop()`.

## Drawing math (`voice/drawing.py`)

Each character lives in a unit box and is a chain of pieces:
- line: `P(t) = P0 + t (P1 − P0)`
- ellipse arc: `P(t) = (cx + rx cos θ(t), cy + ry sin θ(t))`, `θ(t) = a0 + t (a1 − a0)`, `t ∈ [0,1]`

`Page` maps boxes onto the page boundaries, advances a cursor right, and wraps down and back to the left.
Preview all shapes with `python3 drawing.py all`.

## Friction / pressure model (`analysis/`)

```bash
python3 analysis/analyze.py      # uses data/recordings/... by default
python3 analysis/plot.py         # -> docs/friction_analysis.png
```

From the 25 s recording (pen on paper 4.4 s → 21.4 s):

| Quantity | Value |
|---|---|
| Servo pure delay | 67 ms |
| Servo first-order time constant τ | 90 ms |
| Paper plane (follower frame) | z ≈ −32 mm |
| Mean pen drag behind motion, pen down / pen up | 7.3 mm / −0.4 mm |
| Drag vs. pressing depth d | drag ≈ 2.9 mm + 1.6·d (R² 0.32); binned: 0.7 + 2.1·d |
| Drag vs. speed | ~flat (R² 0.02) → Coulomb-dominated, little viscous drag |

Model:
- Servo: `p_cmd(t) = target(t − 67 ms)`, `dp/dt = (p_cmd − p) / 0.09 s`
- Pressure: `F_n = K_z · d`, where `d = max(0, z_paper − z_target)`
- Friction: `F_t = μ F_n`, opposite to pen velocity; data gives `μ · K_z / K_xy ≈ 1.6–2.1`

**Not yet known:** absolute stiffness `K_z`, `K_xy` (N/mm). Measure by pushing the pen tip with a scale; then `μ = 1.6 · K_xy / K_z`.
Until then use μ ≈ 0.3 in simulation.

Caveats: one recording; paper height varies ±6.6 mm (paper not flat or FK error); unexplained −3.9 mm sideways bias.

### MuJoCo mapping (starting point)
- position actuators on each joint; tune `kp` / joint `damping` so the step response matches τ ≈ 90 ms, add ~67 ms command delay
- pen-tip geom vs. paper geom: `friction="0.3 0.005 0.0001"` (replace 0.3 with measured μ)
- contact stiffness via `solref` once `K_z` is measured

## Calibration → joint angles

```
ticks = range_min + (value + 100)/200 · (range_max − range_min)
θ_deg = (ticks − 2048) · 360 / 4096          # relative to the calibration middle pose
```
Follower: shoulder_pan 1.13°/unit, shoulder_lift 1.05, elbow_flex 0.97, wrist_flex 1.00, wrist_roll 1.80.
