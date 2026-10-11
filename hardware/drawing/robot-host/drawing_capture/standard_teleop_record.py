"""LeRobot's teleop loop with seeded startup and passive joint logging."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from corner_capture import make_hardware, port_holders, LEADER_PORT, FOLLOWER_PORT
from lerobot.scripts.lerobot_teleoperate import teleop_loop


def main():
    holders = port_holders((LEADER_PORT, FOLLOWER_PORT))
    if holders:
        raise RuntimeError(f'Serial ports already owned: {holders}')
    leader, follower, processors = make_hardware()
    data = Path(__file__).resolve().parent / 'data'
    data.mkdir(exist_ok=True)
    path = data / ('teleop_demo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.jsonl')
    with path.open('x', buffering=1) as stream:
        try:
            leader.connect(calibrate=False)
            follower.bus.connect()
            if not follower.is_calibrated:
                raise RuntimeError('Follower calibration mismatch')
            initial = follower.bus.sync_read('Present_Position')
            follower.bus.sync_write('Goal_Position', initial)
            follower.configure()
            torque = follower.bus.sync_read('Torque_Enable', normalize=False)
            if not all(value == 1 for value in torque.values()):
                raise RuntimeError(f'Torque enable incomplete: {torque}')
            original_send = follower.send_action
            original_observe = follower.get_observation
            latest = {}
            started = time.monotonic()
            count = 0

            def observe():
                nonlocal latest
                latest = original_observe()
                return latest

            def send(action):
                nonlocal count
                elapsed = time.monotonic() - started
                alpha = min(1.0, elapsed / 5.0)
                # Blend only on startup; no clamp relative to measured position.
                command = {key: initial[key[:-4]] + alpha * (value - initial[key[:-4]])
                           if key.endswith('.pos') else value for key, value in action.items()}
                sent = original_send(command)
                stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(),
                                         'elapsed_s': elapsed, 'follower_observation': latest,
                                         'leader_action': action, 'sent_action': sent}) + '\n')
                count += 1
                if count % 150 == 0:
                    print(f'Teleoperation active: {count} samples, {elapsed:.1f}s', flush=True)
                return sent

            follower.get_observation = observe
            follower.send_action = send
            print(f'TELEOP READY; recording {path}', flush=True)
            teleop_loop(leader, follower, 30, *processors)
        finally:
            # Avoid releasing the follower on a software exit.
            if follower.bus.is_connected:
                follower.bus.disconnect(disable_torque=False)
            if leader.is_connected:
                leader.disconnect()


if __name__ == '__main__':
    main()
