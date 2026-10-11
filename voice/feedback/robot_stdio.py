"""SSH-only JSON RPC. EOF stops the owned robot run; no network listener."""
import argparse
import fcntl
import json
import sys
from pathlib import Path
from controller import Controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical', action='store_true')
    args = parser.parse_args()
    lock = (Path.home() / '.roboarm-voice.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = Controller(physical=args.physical)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                name = request['method']
                if name not in ('draw', 'status', 'stop', 'feedback', 'propose'):
                    raise ValueError('Unknown tool')
                result = getattr(controller, name)(request.get('parameters', {}))
                response = {'result': result}
            except Exception as exc:
                response = {'error': str(exc)}
            print(json.dumps(response), flush=True)
    finally:
        controller.stop()
        if controller.proc is not None:
            try:
                controller.proc.wait(timeout=10)
            except Exception:
                pass

if __name__ == '__main__':
    main()
