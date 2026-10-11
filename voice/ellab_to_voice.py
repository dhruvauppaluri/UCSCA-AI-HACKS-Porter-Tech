"""
friday_voice.py — Voice control for the Friday robot arm.

You talk into the mic -> ElevenLabs agent understands you -> it calls the
`pick_up` or `stop` tool -> this program handles the call.

MODE = "print"   : just prints what Friday would do (use this today, no arm needed)
MODE = "smolvla" : launches the trained SmolVLA policy on the arm (use tomorrow)

Setup:
    pip install "elevenlabs[pyaudio]"
    (Mac:   brew install portaudio  first)
    (Linux: sudo apt install portaudio19-dev  first)

    export ELEVENLABS_API_KEY="your key"
    export ELEVENLABS_AGENT_ID="your agent id"
    python friday_voice.py

The mic used is your computer's DEFAULT input device. Set the USB webcam as
the default microphone in your system sound settings before running.
"""

import os
import signal
import subprocess
import time

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# ---------------------------------------------------------------- settings ---
MODE = "print"  # change to "smolvla" once the model is trained

API_KEY = os.environ["ELEVENLABS_API_KEY"]
AGENT_ID = os.environ["ELEVENLABS_AGENT_ID"]

# Must match EXACTLY the instruction text used when recording the dataset.
TASKS = {
    "lego_brick": "Pick up the lego brick and place it in the bin",
    "pen": "Pick up the pen and place it in the bin",
    "game_piece": "Pick up the game piece and place it in the bin",
}

# Fill these in from Dhruva and Shubh tonight.
ROBOT_PORT = "/dev/ttyACM0"                 # Friday's USB port
ROBOT_ID = "friday"                         # id used during calibration
CAMERA_KEY = "front"                        # must match the name used when recording
CAMERA_INDEX = 0                            # from lerobot-find-cameras opencv
POLICY_PATH = "outputs/train/friday_smolvla/checkpoints/last/pretrained_model"
EPISODE_SECONDS = 30                        # how long Friday gets to finish one pick

# ------------------------------------------------------------- arm control ---
current_run = None  # the running lerobot process, if any


def arm_is_busy():
    return current_run is not None and current_run.poll() is None


def start_policy(task_text):
    """Launch SmolVLA on Friday with the given instruction (non-blocking)."""
    global current_run
    cameras = (
        f'{{ {CAMERA_KEY}: {{type: opencv, index_or_path: {CAMERA_INDEX}, '
        f'width: 640, height: 480, fps: 30, fourcc: "MJPG"}}}}'
    )
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={cameras}",
        "--display_data=false",
        # eval_ prefix + timestamp so each run gets a fresh folder (avoids "File exists")
        f"--dataset.repo_id=local/eval_friday_{int(time.time())}",
        f"--dataset.single_task={task_text}",
        "--dataset.num_episodes=1",
        f"--dataset.episode_time_s={EPISODE_SECONDS}",
        "--dataset.push_to_hub=false",
        f"--policy.path={POLICY_PATH}",
    ]
    print(">> Running:", " ".join(cmd))
    current_run = subprocess.Popen(cmd, start_new_session=True)


def stop_arm():
    global current_run
    if arm_is_busy():
        os.killpg(os.getpgid(current_run.pid), signal.SIGINT)
        print(">> Stop signal sent to Friday")
    current_run = None


# ------------------------------------------------------------ tool handlers ---
def pick_up(parameters):
    item = str(parameters.get("item", "")).strip().lower().replace(" ", "_")
    if item not in TASKS:
        print(f">> Unknown item: {item!r}")
        return f"I don't know the item '{item}'. I can grab the lego brick, pen, or game piece."

    if arm_is_busy():
        return "I'm still busy with the last item. Say stop if you want me to cancel."

    task_text = TASKS[item]
    print(f"\n>> FRIDAY: pick_up({item}) -> \"{task_text}\"")

    if MODE == "smolvla":
        start_policy(task_text)
    return f"Picking up the {item.replace('_', ' ')}."


def stop(parameters=None):
    print("\n>> FRIDAY: STOP")
    if MODE == "smolvla":
        stop_arm()
    return "Stopped."


# ------------------------------------------------------------------- main ---
def main():
    client_tools = ClientTools()
    client_tools.register("pick_up", pick_up)
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

    # Ctrl+C ends the session cleanly (and stops the arm).
    signal.signal(signal.SIGINT, lambda *_: (stop_arm(), conversation.end_session()))

    print(f"Friday is listening (mode: {MODE}). Press Ctrl+C to quit.\n")
    conversation.start_session()
    conversation_id = conversation.wait_for_session_end()
    print(f"\nSession ended. Conversation ID: {conversation_id}")


if __name__ == "__main__":
    main()
