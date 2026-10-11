"""Prepare a shape-regularized nine from the new demonstration; never drive hardware."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent/'smoothed_demo';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'roboalgo'))
from prepare_physical_21 import fk,JOINTS,LOW,HIGH
from z_model import predict_z,rollout
D=np.load(Path(__file__).with_name('demo_arrays.npz'));t=D['time'];observed=D['follower_observation_xyz'];sent=D['sent_action_xyz']
s=json.loads((ROOT/'roboalgo/examples/21/source_sim/summary.json').read_text());R=np.array(s['paper_to_world_rotation']);origin=np.array(s['paper_origin_world_m'])*1000
model=json.loads((ROOT/'roboalgo/artifacts/z_predictor_all_seven.json').read_text())
active=(t>=28)&(t<=53);loop=(t>=36.5)&(t<=41.5);stroke=(t>=36.5)&(t<=50.5)
move=np.r_[False,np.linalg.norm(np.diff(observed[:,:2],axis=0),axis=1)>.1];fitmask=stroke&move
A=np.c_[observed[fitmask,:2],np.ones(fitmask.sum())];surface=np.linalg.lstsq(A,observed[fitmask,2],rcond=None)[0]
surface_rms=float(np.sqrt(np.mean((A@surface-observed[fitmask,2])**2)))
low,high=np.percentile(observed[loop,:2],[5,95],axis=0);center=(low+high)/2;radius=(high-low)/2
end=np.median(observed[(t>=49.5)&(t<=50.5),:2],axis=0)
def ease(u):return u**3*(10-15*u+6*u*u)
time=np.linspace(0,16,481);xyz=[];down=[]
for tt in time:
 if tt<=2:xy=center+np.array([radius[0],0]);lift=20*(1-ease(tt/2));ink=False
 elif tt<=10:
  angle=-2*np.pi*ease((tt-2)/8);xy=center+radius*np.array([np.cos(angle),np.sin(angle)]);lift=0;ink=True
 elif tt<=14:
  u=ease((tt-10)/4);xy=np.array([center[0]+radius[0]+(end[0]-center[0]-radius[0])*u*u,center[1]+(end[1]-center[1])*u]);lift=0;ink=True
 else:xy=end;lift=20*ease((tt-14)/2);ink=False
 xyz.append([*xy,float(np.r_[xy,1]@surface+lift)]);down.append(int(ink))
xyz=np.array(xyz);base=xyz@R.T+origin
q_demo=D['follower_observation'];demo_base=observed@R.T+origin
seed_i=int(np.argmin(np.linalg.norm(demo_base-base[0],axis=1)));q=q_demo[seed_i].copy();commands=[];errors=[]
for target in base:
 for _ in range(100):
  err=target-fk(q)
  if np.linalg.norm(err)<.02:break
  jac=np.column_stack([(fk(q+np.eye(5)[j]*.01)-fk(q-np.eye(5)[j]*.01))/.02 for j in range(5)])
  q=np.clip(q+np.clip(jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(3),err),-2,2),LOW,HIGH)
 commands.append(q.copy());errors.append(float(np.linalg.norm(target-fk(q))))
commands=np.array(commands)
if max(errors)>1:raise RuntimeError('IK residual too large '+str(max(errors)))
raw=[json.loads(l) for l in Path(__file__).with_name('teleop_demo_20261011T180426Z.jsonl').read_text().splitlines()]
grip=float(np.median([r['follower_observation']['gripper.pos'] for r,m in zip(raw,stroke) if m]))
with (OUT/'smoothed_9_joint_candidate_NOT_EXECUTED.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*[n+'.pos' for n in JOINTS],'gripper.pos','pen_down','fk_residual_mm']);w.writerows([tt,*q,grip,d,e] for tt,q,d,e in zip(time,commands,down,errors))
with (OUT/'smoothed_9_paper_waypoints.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s','paper_x_mm','paper_y_mm','paper_z_mm','pen_down']);w.writerows([tt,*p,d] for tt,p,d in zip(time,xyz,down))
sent_base=sent@R.T+origin;ob_base=observed@R.T+origin
pred=predict_z(t[active],sent_base[active],ob_base[active][0,2],model)
report={'recording':'teleop_demo_20261011T180426Z.jsonl','samples':len(t),'recording_duration_s':float(t[-1]),'inferred_loop_window_s':[36.5,41.5],'inferred_contact_window_s':[36.5,50.5],'contact_labels_confirmed':False,'surface_z_from_xy_coefficients':surface.tolist(),'apparent_surface_residual_rms_mm':surface_rms,'loop_center_mm':center.tolist(),'loop_radii_mm':radius.tolist(),'tail_endpoint_mm':end.tolist(),'heldout_z_prediction_rmse_mm':float(np.sqrt(np.mean((pred-ob_base[active,2])**2))),'uncorrected_target_z_rmse_mm':float(np.sqrt(np.mean((sent_base[active,2]-ob_base[active,2])**2))),'candidate_duration_s':16,'candidate_max_ik_error_mm':max(errors),'candidate_max_joint_speed_deg_s':float(np.max(abs(np.diff(commands,axis=0)))*30),'executed':False,'limits':['Coordinates derive from archived FK; no camera or force labels.','The inferred contact surface includes pose-dependent geometry error and is not a measured paper plane.','The intended nine uses the user-specified digit as a shape prior, not automatic recognition.','Predictor was evaluated on the new demonstration, not retrained. Candidate commands have not been compensated or physically validated.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,6))
axes[0].plot(sent[stroke,0],sent[stroke,1],color='#a5a5a5',label='Sent path, reconstructed')
axes[0].plot(observed[stroke,0],observed[stroke,1],color='#c86532',label='Recorded follower, reconstructed')
axes[0].plot(xyz[np.array(down)==1,0],xyz[np.array(down)==1,1],color='#1858a5',lw=2,label='Smoothed 9 candidate')
axes[0].set_aspect('equal');axes[0].set(xlabel='Estimated paper X (mm)',ylabel='Estimated paper Y (mm)',title='Demonstration and shape-regularized candidate');axes[0].legend(fontsize=8,loc='lower left')
axes[1].plot(t[active],sent_base[active,2],color='#aaa',label='Sent target Z')
axes[1].plot(t[active],ob_base[active,2],color='#c86532',label='Recorded follower Z')
axes[1].plot(t[active],pred,color='#1858a5',label='Existing predictor on new recording')
axes[1].set(xlabel='Recording time (s)',ylabel='FK base-frame Z (mm)',title='New demonstration: Z prediction RMSE 3.67 mm');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Offline analysis only — candidate has not been replayed; no image-based pen measurement')
fig.tight_layout();fig.savefig(OUT/'demo_vs_smoothed_9.png',dpi=160)
