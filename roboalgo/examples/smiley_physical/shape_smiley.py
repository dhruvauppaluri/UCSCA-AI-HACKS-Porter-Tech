"""Fit a symmetric smiley to the demonstration; offline, no hardware access."""
import csv,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'even_smiley';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'roboalgo'))
from prepare_physical_21 import fk,JOINTS,LOW,HIGH
D=np.load(HERE/'demo_arrays.npz');t=D['time'];xyz=D['xyz'];qs=D['q']
s=json.loads((ROOT/'roboalgo/examples/21/source_sim/summary.json').read_text());R=np.array(s['paper_to_world_rotation']);origin=np.array(s['paper_origin_world_m'])*1000
left=np.median(xyz[(t>=27.5)&(t<=28.5)],axis=0);right=np.median(xyz[(t>=29.4)&(t<=29.9)],axis=0)
center=(left[:2]+right[:2])/2;half=abs(right[0]-left[0])/2
mouth=xyz[(t>=31.8)&(t<=39)]
radius=np.ptp(mouth[:,0])/2;bottom=np.percentile(mouth[:,1],5);top=np.percentile(mouth[:,1],95);depth=top-bottom
mask=((t>=27.5)&(t<=28.5))|((t>=29.4)&(t<=29.9))|((t>=31.8)&(t<=39))
A=np.c_[xyz[mask,:2],np.ones(mask.sum())];surface=np.linalg.lstsq(A,xyz[mask,2],rcond=None)[0]
def point(x,y,lift=0):return np.array([x,y,np.array([x,y,1])@surface+lift])
def ease(u):return u**3*(10-15*u+6*u*u)
e1=point(center[0]-half,center[1]);e2=point(center[0]+half,center[1]);m1=point(center[0]-radius,top);m2=point(center[0]+radius,top);lift=np.array([0,0,20])
path=[]
def move(a,b,duration,down=0):
 start=path[-1][0] if path else 0
 if not path:path.append([0,*a,down])
 for i in range(1,round(duration*30)+1):
  u=i/round(duration*30);path.append([start+duration*u,*(a+(b-a)*ease(u)),down])
move(e1+lift,e1,1.5);move(e1,e1,.35,1);move(e1,e1+lift,1.5)
move(e1+lift,e2+lift,2);move(e2+lift,e2,1.5);move(e2,e2,.35,1);move(e2,e2+lift,1.5)
move(e2+lift,m1+lift,2);move(m1+lift,m1,1.5)
start=path[-1][0]
for i in range(1,181):
 u=ease(i/180);angle=np.pi*(1-u);p=point(center[0]+radius*np.cos(angle),top-depth*np.sin(angle));path.append([start+6*i/180,*p,1])
move(m2,m2+lift,1.5)
path=np.array(path);q=qs[np.argmin(np.linalg.norm(xyz-path[0,1:4],axis=1))].copy();commands=[];errors=[]
for row in path:
 target=origin+R@row[1:4]
 for _ in range(120):
  err=target-fk(q)
  if np.linalg.norm(err)<.02:break
  J=np.column_stack([(fk(q+np.eye(5)[j]*.01)-fk(q-np.eye(5)[j]*.01))/.02 for j in range(5)])
  q=np.clip(q+np.clip(J.T@np.linalg.solve(J@J.T+.01*np.eye(3),err),-2,2),LOW,HIGH)
 commands.append(q.copy());errors.append(float(np.linalg.norm(target-fk(q))))
assert max(errors)<.1
commands=np.array(commands);speed=np.max(abs(np.diff(commands,axis=0))/np.diff(path[:,0])[:,None]);assert speed<30
raw=[json.loads(l) for l in (HERE/'teleop_demo_20261011T182444Z.jsonl').read_text().splitlines()];grip=float(np.median([r['follower_observation']['gripper.pos'] for r in raw]))
with (OUT/'smiley_joint_candidate.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*[j+'.pos' for j in JOINTS],'gripper.pos','pen_down']);w.writerows([r[0],*q,grip,int(r[-1])] for r,q in zip(path,commands))
with (OUT/'smiley_paper_waypoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s','paper_x_mm','paper_y_mm','paper_z_mm','pen_down']);w.writerows(path)
report={'physically_executed':False,'shape':'Two level stationary dots and one symmetric half-ellipse smile','duration_s':float(path[-1,0]),'eye_spacing_mm':2*half,'mouth_width_mm':2*radius,'mouth_depth_mm':depth,'max_ik_residual_mm':max(errors),'max_joint_speed_deg_s':float(speed),'inferred_contact_surface':surface.tolist(),'contact_surface_rms_mm':float(np.sqrt(np.mean((A@surface-xyz[mask,2])**2))),'limitations':['Stroke windows and paper contact inferred, not force measured.','Symmetry is imposed in model coordinates; real ink symmetry remains unverified.','No 9-specific stretching applied.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6,6));ax.scatter([e1[0],e2[0]],[e1[1],e2[1]],s=100,c='#174b83');arc=path[(path[:,0]>start)&(path[:,0]<=start+6)];ax.plot(arc[:,1],arc[:,2],lw=3,c='#174b83');ax.set(aspect='equal',xlabel='Model paper X (mm)',ylabel='Model paper Y (mm)',title='Planned smiley: two dots and one curve\nPhysical result not yet tested');ax.set_xlim(center[0]-radius-15,center[0]+radius+15);ax.set_ylim(bottom-15,center[1]+15);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'smiley_preview.png',dpi=150)
print(json.dumps(report,indent=2))
