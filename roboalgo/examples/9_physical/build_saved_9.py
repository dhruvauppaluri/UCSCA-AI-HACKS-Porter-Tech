"""Revise only the loop's height using the last physical run; no hardware I/O."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;OUT=HERE/'saved_9';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'roboalgo'));from prepare_physical_21 import fk,JOINTS,LOW,HIGH
s=json.loads((ROOT/'roboalgo/examples/21/source_sim/summary.json').read_text());R=np.array(s['paper_to_world_rotation']);o=np.array(s['paper_origin_world_m'])*1000
report=json.loads((HERE/'smoothed_demo/report.json').read_text());center=np.array(report['loop_center_mm']);surface=np.array(report['surface_z_from_xy_coefficients'])
r=[json.loads(x) for x in (HERE/'smoothed_9_physical_20261011T181325Z.jsonl').read_text().splitlines()];loop=[x for x in r if x['phase']=='smoothed_9' and 9<=x['elapsed_s']<17]
actual=np.array([(fk([x['observed'][n+'.pos'] for n in JOINTS])-o)@R for x in loop]);span=np.ptp(actual[:,:2],axis=0);factor=float(np.clip(span[0]/span[1],1,1.5))
with (HERE/'smoothed_demo/smoothed_9_paper_waypoints.csv').open() as f:rows=list(csv.DictReader(f))
with (HERE/'smoothed_demo/smoothed_9_joint_candidate_NOT_EXECUTED.csv').open() as f:jointrows=list(csv.DictReader(f))
q=np.array([float(jointrows[0][n+'.pos']) for n in JOINTS]);new=[];commands=[];errors=[]
for row in rows:
 t=float(row['time_s']);xyz=np.array([float(row['paper_'+a+'_mm']) for a in 'xyz']);old=xyz.copy()
 if 2<=t<=10:
  xyz[1]=center[1]+factor*(xyz[1]-center[1]);xyz[2]+=surface[1]*(xyz[1]-old[1])
 # Preserve loop; shorten stem by one second at the tested 0.75 playback scale.
 if t>10:
  start=center+np.array([report['loop_radii_mm'][0],0])
  end=np.array(report['tail_endpoint_mm'])
  u=2/3; fraction=u**3*(10-15*u+6*u*u)
  revised=start+(end-start)*np.array([fraction*fraction,fraction])
  delta=revised-end
  if t<=14:
   old_u=(t-10)/4; e=old_u**3*(10-15*old_u+6*old_u*old_u)
   shift=delta*np.array([e*e,e]);t=10+(t-10)*2/3
  else:
   shift=delta;t=t-4/3
  xyz[:2]+=shift;xyz[2]+=surface[:2]@shift
 target=o+R@xyz
 for _ in range(100):
  err=target-fk(q)
  if np.linalg.norm(err)<.02:break
  jac=np.column_stack([(fk(q+np.eye(5)[j]*.01)-fk(q-np.eye(5)[j]*.01))/.02 for j in range(5)])
  q=np.clip(q+np.clip(jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(3),err),-2,2),LOW,HIGH)
 errors.append(float(np.linalg.norm(target-fk(q))));commands.append(q.copy());new.append([t,*xyz,int(row['pen_down'])])
if max(errors)>1:raise RuntimeError('IK failure')
with (OUT/'nine_paper_waypoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s','paper_x_mm','paper_y_mm','paper_z_mm','pen_down']);w.writerows(new)
with (OUT/'nine_joint_commands.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*[n+'.pos' for n in JOINTS],'gripper.pos','pen_down','fk_residual_mm']);w.writerows([p[0],*q,float(j['gripper.pos']),p[-1],e] for p,q,j,e in zip(new,commands,jointrows,errors))
summary={'source_physical_log':'smoothed_9_physical_20261011T181325Z.jsonl','observed_loop_xy_span_mm':span.tolist(),'loop_y_command_multiplier':factor,'duration_s':44/3,'tail_paper_path_unchanged':False,'stem_shortened_playback_seconds':1,'expected_total_seconds':23.4,'source_rounder_run_user_feedback':'basically perfect(ish)','shortened_version_physically_tested':False,'max_ik_residual_mm':max(errors),'max_joint_speed_deg_s':float(np.max(np.abs(np.diff(np.array(commands),axis=0)))*30),'executed':False,'note':'Empirical axis scaling from one servo-reconstructed run. It is not proof of a circular physical ink result; hysteresis/contact/geometry remain uncertain.'}
(OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
