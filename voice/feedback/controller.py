"""Persistent, bounded feedback workflow. No microphone or hardware imports."""
import csv, hashlib, json, math, os, signal, subprocess, sys, threading, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCES={
 '9':ROOT/'roboalgo/examples/9_physical/saved_9',
 'smiley':ROOT/'roboalgo/examples/smiley_physical/smiley_face',
}
FILES={'9':('nine_joint_commands.csv','nine_paper_waypoints.csv','draw_9.py'), 'smiley':('high_lift_smiley.csv','smiley_paper_waypoints.csv','smiley_face.py')}
LIMITS={'lift_delta_mm':(0,10),'stem_scale':(.8,1.0),'mouth_depth_scale':(.85,1.15),'eye_spacing_scale':(.85,1.15)}

class Controller:
 def __init__(self, directory=None, physical=False):
  self.home=Path(directory or ROOT/'runs/voice_feedback');self.home.mkdir(parents=True,exist_ok=True)
  self.physical=physical;self.lock=threading.RLock();self.proc=None
  p=self.home/'state.json'
  self.state=json.loads(p.read_text()) if p.exists() else {'preferred':{},'revisions':{},'runs':{},'requests':{}}
  for run in self.state['runs'].values():
   if run['status'] in ('running','stopping'):run['status']='interrupted_unknown'
  self.save()
 def save(self):
  tmp=self.home/'state.tmp';tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.home/'state.json')
 def routine(self,name):
  name=str(name).strip().lower()
  name={'nine':'9','smiley face':'smiley','smile':'smiley'}.get(name,name)
  if name not in SOURCES:raise ValueError('Supported routines: 9, smiley face')
  return name
 def baseline(self,name):
  key=name+'-baseline'
  if key not in self.state['revisions']:
   self.state['revisions'][key]={'routine':name,'directory':str(SOURCES[name]),'files':FILES[name],'reward':0,'baseline':True}
  return key
 def status(self,parameters=None):
  with self.lock:
   return {'physical_mode':self.physical,'busy':self.proc is not None and self.proc.poll() is None,'runs':list(self.state['runs'].values())[-5:],'preferred':self.state['preferred']}
 def draw(self,p):
  with self.lock:
   request=str(p.get('request_id','')).strip()
   if not request:raise ValueError('request_id required for retry-safe execution')
   if request in self.state['requests']:return self.state['runs'][self.state['requests'][request]]
   if self.proc is not None and self.proc.poll() is None:raise ValueError('Arm busy')
   name=self.routine(p['content']);rev=p.get('revision_id') or self.state['preferred'].get(name) or self.baseline(name)
   if rev not in self.state['revisions'] or self.state['revisions'][rev]['routine']!=name:raise ValueError('Wrong revision')
   spec=self.state['revisions'][rev];folder=Path(spec['directory']);csvfile=folder/spec['files'][0]
   rows=list(csv.DictReader(csvfile.read_text().splitlines()));times=[float(r['time_s']) for r in rows]
   if len(rows)<2 or not all(math.isfinite(float(v)) for r in rows for v in r.values()) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Invalid trajectory')
   ident=uuid.uuid4().hex;run={'run_id':ident,'revision_id':rev,'routine':name,'physical':self.physical,'status':'running' if self.physical else 'dry_run','csv_sha256':hashlib.sha256(csvfile.read_bytes()).hexdigest()}
   self.state['runs'][ident]=run;self.state['requests'][request]=ident;self.save()
   if self.physical:
    logpath=self.home/(ident+'.log')
    try:
     with logpath.open('w') as log:self.proc=subprocess.Popen([sys.executable,str(folder/spec['files'][2])],cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except Exception:
     run['status']='failed';self.save();raise
    threading.Thread(target=self._wait,args=(self.proc,ident,logpath),daemon=True).start()
   return dict(run)
 def _wait(self,proc,ident,log):
  code=proc.wait()
  with self.lock:
   run=self.state['runs'][ident]
   if run['status']=='running':run['status']='completed' if code==0 and 'SECONDS; HOLDING; LOG' in log.read_text() else 'failed'
   if run['status']=='stopping':run['status']='stopped'
   run['exit_code']=code;run['log']=str(log);self.save()
 def stop(self,p=None):
  with self.lock:
   if self.proc is not None and self.proc.poll() is None:
    os.killpg(self.proc.pid,signal.SIGINT)
    for r in self.state['runs'].values():
     if r['status']=='running':r['status']='stopping'
    self.save();return {'status':'stop_requested','message':'Waiting for controller to stop; no new drawing while process remains active.'}
   return {'status':'idle'}
 def feedback(self,p):
  with self.lock:
   run=self.state['runs'][p['run_id']]
   if run['status']!='completed' or not run['physical']:raise ValueError('Rate only a completed physical run')
   if 'rating' in run:raise ValueError('Run already rated')
   accepted=p.get('accepted')
   if type(accepted) is not bool:raise ValueError('accepted must be boolean')
   run.update(rating=1 if accepted else -1,feedback=str(p.get('feedback','')))
   revision=self.state['revisions'][run['revision_id']];revision['reward']+=run['rating']
   if accepted:self.state['preferred'][run['routine']]=run['revision_id']
   elif self.state['preferred'].get(run['routine'])==run['revision_id']:del self.state['preferred'][run['routine']]
   self.save();return {'saved':True,'reward':run['rating'],'next':'Keep approved revision' if accepted else 'Propose a bounded correction; do not automatically replay'}
 def propose(self,p):
  from planner import build
  with self.lock:
   run=self.state['runs'][p['run_id']]
   if run.get('rating')!=-1:raise ValueError('Record negative feedback for this run first')
   changes=p['changes']
   if not isinstance(changes,dict) or not changes:raise ValueError('changes must be a nonempty object')
   for key,value in changes.items():
    if key not in LIMITS or type(value) not in (int,float) or not math.isfinite(value) or not LIMITS[key][0]<=value<=LIMITS[key][1]:raise ValueError('Unsupported or out-of-bounds correction: '+key)
    if key=='stem_scale' and run['routine']!='9':raise ValueError('Stem applies only to 9')
    if key in ('mouth_depth_scale','eye_spacing_scale') and run['routine']!='smiley':raise ValueError('Face correction applies only to smiley')
   ident=uuid.uuid4().hex;folder=self.home/ident
   # Always relative to the frozen baseline: bounds cannot accumulate across retries.
   base=self.state['revisions'][self.baseline(run['routine'])]
   report=build(base,folder,changes)
   self.state['revisions'][ident]={'routine':run['routine'],'directory':str(folder),'files':base['files'],'reward':0,'baseline':False,'changes':changes,'parent_run':run['run_id'],'report':report}
   self.save();return {'revision_id':ident,'changes':changes,'report':report,'status':'prepared_not_executed'}
