"""Mac audio + ElevenLabs client tools; robot controller stays on AMD via SSH."""
import argparse
import json
import os
import signal
import subprocess
import threading

class RemoteController:
    def __init__(self, host, socket, physical=False):
        command = 'cd ~/roboarm-voice && ~/miniforge3/envs/lerobot/bin/python -u voice/feedback/robot_stdio.py'
        if physical:
            command += ' --physical'
        self.proc = subprocess.Popen(['ssh', '-S', socket, '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=3', host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.lock = threading.Lock()
    def call(self, method, parameters=None):
        with self.lock:
            self.proc.stdin.write(json.dumps({'method': method, 'parameters': parameters or {}}) + '\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError('AMD connection closed')
            result = json.loads(line)
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result['result']
    def close(self):
        try:
            self.call('stop')
        finally:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--host', default=os.environ.get('AMD_SSH_TARGET', 'amd-genai2'))
    parser.add_argument('--socket', default='/tmp/roboarm-mac-ssh')
    args = parser.parse_args()
    remote = RemoteController(args.host, args.socket, args.physical)
    ended = threading.Event()
    conversation = None
    try:
        initial = remote.call('status')
        print(json.dumps(initial, indent=2), flush=True)
        if args.check:
            return
        from elevenlabs import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
        from mac_audio import MacAudioInterface
        tools = ClientTools()
        def wrap(name):
            def call(parameters):
                try:
                    return json.dumps(remote.call(name, parameters))
                except Exception as exc:
                    return json.dumps({'error': str(exc)})
            return call
        for name in ('draw', 'status', 'stop', 'feedback', 'propose'):
            tools.register(name, wrap(name))
        conversation = Conversation(client=ElevenLabs(api_key=os.environ['ELEVENLABS_API_KEY']), agent_id=os.environ['ELEVENLABS_AGENT_ID'], requires_auth=True, audio_interface=MacAudioInterface(), client_tools=tools, callback_user_transcript=lambda text: print('You:', text, flush=True), callback_agent_response=lambda text: print('Friday:', text, flush=True))
        def stop(*_):
            ended.set()
            conversation.end_session()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        def notify():
            seen = {r['run_id'] for r in initial['runs']}
            while not ended.wait(1):
                try:
                    for run in remote.call('status')['runs']:
                        if run['run_id'] not in seen and run['status'] in ('completed', 'failed', 'stopped'):
                            conversation.send_user_message('Application robot-status event (not user feedback): ' + json.dumps(run) + '. If completed, ask whether the drawing looked right; otherwise report the failure or stop.')
                            seen.add(run['run_id'])
                except Exception as exc:
                    print('Robot connection error:', exc, flush=True)
                    conversation.end_session()
                    return
        conversation.start_session()
        threading.Thread(target=notify, daemon=True).start()
        conversation.wait_for_session_end()
    finally:
        ended.set()
        if conversation:
            conversation.end_session()
        remote.close()

if __name__ == '__main__':
    main()
