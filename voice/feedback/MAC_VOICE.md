# Mac voice, AMD robot

The Mac uses the **EarPods Microphone** and **MacBook Pro Speakers**, explicitly selected by device name. The AMD machine executes the existing saved 9 and smiley routines. Five client tools (`draw`, `status`, `stop`, `feedback`, `propose`) travel through an authenticated SSH connection. No OpenAI credits or public webhook are needed.

## Start on this Mac

From this repository:

```bash
# Connection check only; no microphone session or robot motion
./voice/feedback/start_mac_voice.sh --check

# Voice conversation, simulated drawing commands
./voice/feedback/start_mac_voice.sh

# Voice conversation with actual saved robot routines
./voice/feedback/start_mac_voice.sh --physical
```

Say “draw a smiley face” or “draw nine.” Ctrl+C ends the conversation and requests a stop on the AMD controller. A failed SSH connection ends the server and requests a stop; disconnect detection is not an instantaneous emergency stop.

Keep the EarPods connected to the Mac. Grant microphone access to the launching app if macOS asks. The client refuses to silently fall back to other devices.

## Installed setup

- Local Python: `work/sim-env` beside the repository, with ElevenLabs and PyAudio installed.
- Private Mac credentials: `~/.config/roboarm/mac-voice.env`, permissions 600; only ElevenLabs credentials copied from AMD.
- SSH target: set via `AMD_SSH_TARGET` (user@host) in your private env file; authenticate with an SSH key, and never commit passwords.
- AMD server: `~/roboarm-voice/voice/feedback/robot_stdio.py`.
- Server holds a lock to reject another instance of this SSH server. Do not run the older standalone AMD voice bridge at the same time.
- Existing AMD state preserves approved routines and ratings.
- `ROBOARM_VOICE_PYTHON` overrides the launcher’s Python path for another Mac.

## Verified during setup

Agent API access; SSH status; dry-run 9 and smiley dispatch; duplicate-request protection; idle stop; named audio devices and supported 16 kHz mono formats. Live Mac microphone capture and audible replies still need the first voice session. No physical drawing was performed during setup.
