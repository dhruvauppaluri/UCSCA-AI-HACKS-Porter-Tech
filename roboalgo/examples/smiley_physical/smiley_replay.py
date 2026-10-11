"""Smoothed demonstration replay for the calibrated AMD station. Importing does not move hardware."""
def draw_smiley():
 import sys,json,time,csv,math
 from pathlib import Path
 from datetime import datetime,timezone
 sys.path.insert(0,str(Path.home()/'drawing_capture'))
 from corner_capture import port_holders
 sys.path.insert(0,str(Path.home()/'usefulcommands'))
 from play_dataforarm import load_trajectory
 from lerobot.robots.so_follower import SOFollower,SOFollowerRobotConfig
 port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C4C128167-if00'
 if port_holders((port,)):raise RuntimeError('Follower serial port already owned')
 p=Path(__file__).with_name('smiley_smoothed_joint_candidate.csv')
 plan,duration=load_trajectory(p)
 with p.open() as f:meta=list(csv.DictReader(f))
 if len(plan)!=1516 or abs(duration-51.470532263978384)>1e-5:raise RuntimeError('Unexpected trajectory')
 names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
 def gap(a,b):return max(abs(a[n+'.pos']-b[n+'.pos']) for n in names)
 r=SOFollower(SOFollowerRobotConfig(port=port,id='my_awesome_follower_arm',use_degrees=True,disable_torque_on_disconnect=False))
 logpath=Path.home()/'drawing_capture/data'/('smiley_physical_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.jsonl')
 r.bus.connect()
 try:
  if not r.is_calibrated:raise RuntimeError('Calibration mismatch')
  modes=r.bus.sync_read('Operating_Mode',normalize=False)
  if any(v!=0 for v in modes.values()):raise RuntimeError('Not in position mode: '+str(modes))
  initial=r.get_observation();first=plan[0][1];keys=list(first)
  if gap(initial,first)>145:raise RuntimeError('Unexpected initial pose')
  r.send_action({k:initial[k] for k in keys});r.bus.enable_torque()
  if not all(v==1 for v in r.bus.sync_read('Torque_Enable',normalize=False).values()):raise RuntimeError('Torque enable failed')
  last={k:initial[k] for k in keys};bad_since=None;began=time.monotonic();report_at=0.
  with logpath.open('x',buffering=1) as log:
   def command(phase,target,contact=False):
    nonlocal last,bad_since,report_at
    actual=r.get_observation();now=time.monotonic();err=gap(actual,last)
    if now-began>duration+15:raise RuntimeError('Replay deadline')
    limit=10 if contact else 15
    if err>limit:
     if bad_since is None:bad_since=now
    else:bad_since=None
    if err>25 or (bad_since is not None and now-bad_since>.5):raise RuntimeError('Tracking failure '+str(err))
    sent=r.send_action(target);last=sent.copy()
    log.write(json.dumps({'elapsed_s':now-began,'phase':phase,'target':target,'observed':actual,'sent':sent,'tracking_gap_deg':err})+'\n')
    if now-report_at>2:
     print(phase,round(now-began,1),'s; tracking',round(err,2),'deg',flush=True);report_at=now
   command('hold_start',last)
   for i in range(1,181):
    f=i/180;b=f*f*(3-2*f)
    time.sleep(max(0,began+6*f-time.monotonic()))
    command('approach',{k:initial[k]+b*(first[k]-initial[k]) for k in keys})
   print('START_SMILEY',flush=True)
   for (t,target),m in zip(plan,meta):
    time.sleep(max(0,began+6+t-time.monotonic()))
    command('smoothed_smiley',target,True)
   print('SMILEY_COMPLETE; RETURNING_TO_REST',flush=True)
   rest={'shoulder_pan.pos': 5.714285714285714, 'shoulder_lift.pos': -98.10989010989012, 'elbow_flex.pos': 95.2967032967033, 'wrist_flex.pos': 56.967032967032964, 'wrist_roll.pos': 16.747252747252748, 'gripper.pos': 1.8060200668896322}
   back=r.get_observation()
   for i in range(1,181):
    f=i/180;b=f*f*(3-2*f)
    time.sleep(max(0,began+6+duration+6*f-time.monotonic()))
    command('return_rest',{k:back[k]+b*(rest[k]-back[k]) for k in keys})
   for _ in range(12):
    command('rest_hold',{k:rest[k] for k in keys});time.sleep(1/30)
   print('REST_ERROR_DEG',round(gap(r.get_observation(),rest),2),flush=True)
   print('COMPLETE_SMILEY_AND_RETURN; ELAPSED',round(time.monotonic()-began,2),'SECONDS; HOLDING; LOG',logpath,flush=True)
 except BaseException as exc:
  print('STOPPED',type(exc).__name__,str(exc),flush=True)
  try:
   actual=r.get_observation();r.send_action({k:actual[k] for k in plan[0][1]})
  except Exception as e:print('HOLD ERROR',str(e),flush=True)
  raise
 finally:r.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
 draw_smiley()
