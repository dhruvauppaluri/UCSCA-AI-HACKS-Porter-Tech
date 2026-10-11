"""Exercise the deployed playback parser and loop without connecting hardware."""

import contextlib
import csv
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "hardware/drawing/robot-host/usefulcommands/play_dataforarm.py"
PLAN = ROOT / "roboalgo/examples/21/physical_hover_21_PLAYBACK.csv"


class PlaybackCompatibility(unittest.TestCase):
    def load_script(self, robot_class=object):
        module = types.ModuleType("lerobot.robots.so_follower")
        module.SOFollower = robot_class
        module.SOFollowerRobotConfig = lambda **kw: types.SimpleNamespace(**kw)
        modules = {"lerobot": types.ModuleType("lerobot"),
                   "lerobot.robots": types.ModuleType("lerobot.robots"),
                   "lerobot.robots.so_follower": module}
        spec = importlib.util.spec_from_file_location("original_playback", SCRIPT)
        script = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(script)
        return script

    def test_original_parser_accepts_hover_without_changing_body_targets(self):
        script = self.load_script()
        trajectory, duration = script.load_trajectory(PLAN)
        with (PLAN.parent / "physical_hover_21_FOR_REVIEW.csv").open(newline="") as handle:
            reference = list(csv.DictReader(handle))
        self.assertEqual(len(trajectory), len(reference))
        self.assertAlmostEqual(duration, 17.6)
        for (timestamp, action), row in zip(trajectory, reference):
            self.assertEqual(set(action), set(script.JOINT_COLUMNS))
            self.assertEqual(timestamp, float(row["time_s"]))
            for name in script.JOINT_COLUMNS[:-1]:
                self.assertEqual(action[name], float(row[name]))
            self.assertAlmostEqual(action["gripper.pos"], 2.0249221183800623)

    def test_original_loop_sends_every_row_at_requested_speed(self):
        clock = types.SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now
        def sleep(seconds):
            self.assertGreaterEqual(seconds, 0)
            clock.now += seconds
        clock.sleep = sleep
        calls = []
        class Robot:
            def __init__(self, config):
                calls.append(("config", config))
            def connect(self):
                calls.append(("connect", clock.now))
            def send_action(self, action):
                calls.append(("send", clock.now, action.copy()))
            def disconnect(self):
                calls.append(("disconnect", clock.now))
        script = self.load_script(Robot)
        script.time = clock
        with patch.object(sys, "argv", [str(SCRIPT), str(PLAN), "--speed", "0.4"]), \
                contextlib.redirect_stdout(io.StringIO()):
            script.main()
        sends = [c for c in calls if c[0] == "send"]
        trajectory, _ = script.load_trajectory(PLAN)
        self.assertEqual(len(sends), len(trajectory))
        for (_, sent_at, sent), (timestamp, expected) in zip(sends, trajectory):
            self.assertEqual(sent, expected)
            self.assertAlmostEqual(sent_at, 3.0 + timestamp / 0.4)
        self.assertEqual(calls[-1][0], "disconnect")
        self.assertAlmostEqual(clock.now, 47.0)


if __name__ == "__main__":
    unittest.main()
