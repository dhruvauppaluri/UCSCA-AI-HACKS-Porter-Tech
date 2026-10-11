"""
friday_voice.py — Voice control for the Friday drawing arm.

"Friday, draw 42"            -> draws 4 then 2 at the cursor, cursor moves right
"Friday, draw a smiley face" -> draws a smiley at the cursor, cursor moves right
At the right edge of the page the cursor moves DOWN a line and back to the LEFT.
"Friday, new page"           -> cursor back to the top-left (after you swap the paper)
"Friday, stop"               -> stops the arm

ARM_MODE = "print": saves path.json + preview.png, prints the strokes (no arm needed)
ARM_MODE = "real" : also calls send_to_arm()  (Dhruva & Shubh fill that in)

Setup (Mac):
    brew install portaudio
    python3.15 -m pip install "elevenlabs[pyaudio]" matplotlib
    export ELEVENLABS_API_KEY="..."
    export ELEVENLABS_AGENT_ID="..."
    python3.15 friday_voice.py

Keep drawing.py in the same folder as this file.
"""

import json
import os
import signal
import threading

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

import drawing

# ---------------------------------------------------------------- settings ---
ARM_MODE = "print"   # change to "real" once send_to_arm() drives the arm

API_KEY = os.environ["ELEVENLABS_API_KEY"]
AGENT_ID = os.environ["ELEVENLABS_AGENT_ID"]

# Page boundaries in the ARM's coordinate system (get these from Dhruva & Shubh,
# e.g. centimeters measured from the arm's base). y_max is the TOP of the page.
PAGE = {"x_min": 0.0, "x_max": 20.0, "y_min": 0.0, "y_max": 15.0}
CHAR_HEIGHT = 3.0     # height of one digit, same units as PAGE
MARGIN = 0.5          # blank border inside the page
LINE_GAP = 0.4        # space between lines, as a fraction of CHAR_HEIGHT

page = drawing.Page(PAGE, CHAR_HEIGHT, line_gap=LINE_GAP, margin=MARGIN)

# ------------------------------------------------------------- arm control ---
stop_event = threading.Event()
arm_thread = None


def arm_busy():
    return arm_thread is not None and arm_thread.is_alive()


def send_to_arm(strokes):
    """
    >>> Dhruva & Shubh: replace the body of this function. <<<

    `strokes` is a list of strokes. Each stroke is a list of (x, y) points in
    PAGE coordinates. For each stroke:
        1. pen UP, move to stroke[0]
        2. pen DOWN
        3. move through every point in order
    After the last stroke: pen UP (stay above the page, ready for the next one).
    Check `stop_event.is_set()` often and quit immediately if it is.
    """
    for i, stroke in enumerate(strokes, 1):
        if stop_event.is_set():
            print(">> Drawing stopped.")
            return
        print(f"   stroke {i}/{len(strokes)}: pen up -> {fmt(stroke[0])}, "
              f"pen down, {len(stroke)} points -> {fmt(stroke[-1])}")
    print(">> Done (simulated). Pen up.")


def fmt(p):
    return f"({p[0]:.1f}, {p[1]:.1f})"


# ------------------------------------------------------------ tool handlers ---
def draw(parameters):
    global arm_thread
    content = str(parameters.get("content", "")).strip()

    if arm_busy():
        return "I'm still drawing. Say stop to cancel."

    already_on_page = list(page.drawn)
    try:
        strokes = page.place(content)
    except drawing.PageFull:
        print(">> Page is full")
        return "The page is full. Put in a new page and say new page."
    except ValueError:
        print(f">> Can't draw {content!r}")
        return f"I can't draw '{content}'. I can draw letters, digits, spaces, dots, dashes, or a smiley face."

    print(f"\n>> FRIDAY: draw({content}) -> {len(strokes)} strokes. "
          f"Cursor now x={page.x:.1f}, line top y={page.y_top:.1f}")

    with open("path.json", "w") as f:
        json.dump({"content": content, "page": PAGE, "strokes": strokes}, f)
    try:
        drawing.save_preview(strokes, PAGE, "preview.png", already_drawn=already_on_page)
        print(">> Saved path.json and preview.png")
    except ImportError:
        print(">> Saved path.json (pip install matplotlib for preview.png)")

    stop_event.clear()
    if ARM_MODE == "real":
        arm_thread = threading.Thread(target=send_to_arm, args=(strokes,), daemon=True)
        arm_thread.start()
    else:
        send_to_arm(strokes)

    label = "a smiley face" if drawing.tokens_for(content) == ["smiley"] else content
    return f"Drawing {label}."


def new_page(parameters=None):
    if arm_busy():
        return "I'm still drawing. Say stop first."
    page.new_page()
    print("\n>> FRIDAY: NEW PAGE (cursor back to top-left)")
    return "Ready on a new page."


def stop(parameters=None):
    print("\n>> FRIDAY: STOP")
    stop_event.set()
    return "Stopped."


# ------------------------------------------------------------------- main ---
def main():
    client_tools = ClientTools()
    client_tools.register("draw", draw)
    client_tools.register("new_page", new_page)
    client_tools.register("stop", stop)

    conversation = Conversation(
        client=ElevenLabs(api_key=API_KEY),
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=client_tools,
        callback_user_transcript=lambda text: print(f"You:    {text}"),
        callback_agent_response=lambda text: print(f"Friday: {text}"),
    )

    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), conversation.end_session()))

    print(f"Friday is listening (arm mode: {ARM_MODE}). Press Ctrl+C to quit.\n")
    conversation.start_session()
    conversation.wait_for_session_end()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
