"""
friday_voice.py — say "Friday, draw 42" or "Friday, draw 9" and the follower arm draws it.

How it works:
  mic -> ElevenLabs agent (Friday) -> calls the `draw` or `stop` tool
      -> this program runs (or stops) the saved drawing script for that number

Setup (on the laptop plugged into the arm):
    pip install "elevenlabs[pyaudio]"        # Mac: brew install portaudio first
    export ELEVENLABS_API_KEY="..."
    export ELEVENLABS_AGENT_ID="..."
    python3 friday_voice.py

ElevenLabs agent needs two CLIENT tools:
    draw  (string parameter `content`)   and   stop  (no parameters)
"""

import os
import shlex
import signal
import subprocess

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# ============ EDIT THIS ============
# What Friday can draw -> the exact command that makes the follower draw it.
# Add more lines to add more drawings.
DRAWINGS = {
    "42": "python3 draw_42.py",
    "9":  "python3 draw_9.py",
}
# Folder the drawing scripts live in (default: same folder as this file)
DRAW_FOLDER = os.path.dirname(os.path.abspath(__file__))
# ===================================

API_KEY = os.environ["ELEVENLABS_API_KEY"]
AGENT_ID = os.environ["ELEVENLABS_AGENT_ID"]

arm_proc = None   # the running drawing script, if any


def arm_busy():
    return arm_proc is not None and arm_proc.poll() is None


WORDS = {"forty two": "42", "forty-two": "42", "fourtytwo": "42", "nine": "9"}


def draw(parameters):
    global arm_proc
    content = str(parameters.get("content", "")).strip().lower()
    content = WORDS.get(content, content).replace(" ", "")
    if content not in DRAWINGS:
        options = " or ".join(DRAWINGS)
        return f"I can only draw {options} right now."
    if arm_busy():
        return "I'm still drawing. Say stop to cancel."
    cmd = DRAWINGS[content]
    print(f"\n>> FRIDAY: drawing {content}  ->  {cmd}   (in {DRAW_FOLDER})")
    arm_proc = subprocess.Popen(shlex.split(cmd), cwd=DRAW_FOLDER, start_new_session=True)
    return f"Drawing {content}."


def stop(parameters=None):
    global arm_proc
    print("\n>> FRIDAY: STOP")
    if arm_busy():
        os.killpg(os.getpgid(arm_proc.pid), signal.SIGINT)   # same as pressing Ctrl+C on the script
    arm_proc = None
    return "Stopped."


def main():
    missing = [c.split()[-1] for c in DRAWINGS.values()
               if c.split()[-1].endswith(".py") and not os.path.exists(os.path.join(DRAW_FOLDER, c.split()[-1]))]
    if missing:
        print(f"WARNING: not found in {DRAW_FOLDER}: {', '.join(missing)}\n")

    tools = ClientTools()
    tools.register("draw", draw)
    tools.register("stop", stop)

    conversation = Conversation(
        client=ElevenLabs(api_key=API_KEY),
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=tools,
        callback_user_transcript=lambda text: print(f"You:    {text}"),
        callback_agent_response=lambda text: print(f"Friday: {text}"),
    )

    # Ctrl+C: stop the arm first, then end the voice session
    signal.signal(signal.SIGINT, lambda *_: (stop(), conversation.end_session()))

    print(f"Friday is listening. Can draw: {', '.join(DRAWINGS)}. Ctrl+C to quit.\n")
    conversation.start_session()
    conversation.wait_for_session_end()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
