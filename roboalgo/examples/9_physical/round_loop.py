"""Revise only the loop's height using the last physical run; no hardware I/O."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;OUT=HERE/'rounder_loop';OUT.mkdir(exist_ok=True)
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
 target=o+R@xyz
 for _ in range(100):
  err=target-fk(q)
  if np.linalg.norm(err)<.02:break
  jac=np.column_stack([(fk(q+np.eye(5)[j]*.01)-fk(q-np.eye(5)[j]*.01))/.02 for j in range(5)])
  q=np.clip(q+np.clip(jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(3),err),-2,2),LOW,HIGH)
 errors.append(float(np.linalg.norm(target-fk(q))));commands.append(q.copy());new.append([t,*xyz,int(row['pen_down'])])
if max(errors)>1:raise RuntimeError('IK failure')
with (OUT/'rounder_9_paper_waypoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s','paper_x_mm','paper_y_mm','paper_z_mm','pen_down']);w.writerows(new)
with (OUT/'rounder_9_joint_candidate_NOT_EXECUTED.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*[n+'.pos' for n in JOINTS],'gripper.pos','pen_down','fk_residual_mm']);w.writerows([p[0],*q,float(j['gripper.pos']),p[-1],e] for p,q,j,e in zip(new,commands,jointrows,errors))
summary={'source_physical_log':'smoothed_9_physical_20261011T181325Z.jsonl','observed_loop_xy_span_mm':span.tolist(),'loop_y_command_multiplier':factor,'duration_s':16,'tail_paper_path_unchanged':True,'max_ik_residual_mm':max(errors),'max_joint_speed_deg_s':float(np.max(np.abs(np.diff(np.array(commands),axis=0)))*30),'executed':False,'note':'Empirical axis scaling from one servo-reconstructed run. It is not proof of a circular physical ink result; hysteresis/contact/geometry remain uncertain.'}
(OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,7));a=np.array(new);mask=(a[:,0]>=2)&(a[:,0]<=10);old=np.array([[float(r['paper_x_mm']),float(r['paper_y_mm'])] for r in rows])
ax.plot(actual[:,0],actual[:,1],color='#c76b32',label='Last recorded loop (FK estimate)')
ax.plot(old[mask,0],old[mask,1],color='#888',ls='--',label='Previous commanded loop')
ax.plot(a[mask,1],a[mask,2],color='#2456a6',label='Revised command: height +41%')
ax.set(aspect='equal',xlabel='Estimated paper X (mm)',ylabel='Estimated paper Y (mm)',title='Taller command to compensate for measured flattening\nRevised path has not been executed')
ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(OUT/'loop_adjustment.png',dpi=150)
