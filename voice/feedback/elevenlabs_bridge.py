"""Run locally on the robot station; default mode never moves hardware."""
import argparse,json,os,signal,threading
from controller import Controller

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');parser.add_argument('--adjustment-api',action='store_true');parser.add_argument('--adjustment-port',type=int,default=8780);args=parser.parse_args()
 from elevenlabs import ElevenLabs
 from elevenlabs.conversational_ai.conversation import Conversation,ClientTools
 from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
 controller=Controller(physical=args.physical);tools=ClientTools();server=None
 from openai_adjustment import Adjuster
 adjuster=Adjuster(controller)
 if args.adjustment_api:
  from adjustment_http import start_server
  server=start_server(adjuster,os.environ.get('ADJUSTMENT_API_TOKEN'),port=args.adjustment_port)
 def wrap(fn):
  def call(p):
   try:return json.dumps(fn(p))
   except Exception as exc:return json.dumps({'error':str(exc)})
  return call
 for name in ('draw','status','feedback','propose','stop'):tools.register(name,wrap(getattr(controller,name)))
 tools.register('request_adjustment',wrap(adjuster.request))
 conversation=Conversation(client=ElevenLabs(api_key=os.environ['ELEVENLABS_API_KEY']),agent_id=os.environ['ELEVENLABS_AGENT_ID'],requires_auth=True,audio_interface=DefaultAudioInterface(),client_tools=tools,callback_user_transcript=lambda text:print('You:',text),callback_agent_response=lambda text:print('Friday:',text))
 signal.signal(signal.SIGINT,lambda *_:(controller.stop(),conversation.end_session()))
 ended=threading.Event()
 def notify_completed():
  notified=set(controller.state['runs'])
  while not ended.wait(1):
   for run in controller.status()['runs']:
    if run['run_id'] not in notified and run['status'] in ('completed','failed','stopped'):
     try:
      conversation.send_user_message('Application robot-status event (not user feedback): '+json.dumps(run)+'. If completed, ask the user whether the drawing looked right. Otherwise report the stop/failure.')
      notified.add(run['run_id'])
     except Exception as exc:print('Completion notification pending:',exc)
 try:
  conversation.start_session()
  threading.Thread(target=notify_completed,daemon=True).start()
  conversation.wait_for_session_end()
 finally:
  ended.set();controller.stop()
  if server:server.shutdown();server.server_close()
if __name__=='__main__':main()
