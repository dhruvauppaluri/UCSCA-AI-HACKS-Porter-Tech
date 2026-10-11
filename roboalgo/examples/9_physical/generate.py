import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'roboalgo'))
from prepare_physical_21 import fk,JOINTS,LOW,HIGH
out=Path(__file__).parent
s=json.loads((ROOT/'roboalgo/examples/21/source_sim/summary.json').read_text())
R=np.array(s['paper_to_world_rotation']); origin=np.array(s['paper_origin_world_m'])*1000
with (ROOT/'roboalgo/examples/21/physical_hover_21_PLAYBACK.csv').open() as f: first=next(csv.DictReader(f))
q=np.array([float(first[n+'.pos']) for n in JOINTS]); rows=[]; errors=[]
def smooth(u): return u*u*u*(10+u*(-15+6*u))
for t in np.linspace(0,8,241):
 if t<=1.5: xyz=np.array([28.,216.,30*(1-smooth(t/1.5))]); down=0
 elif t<=5:
  theta=-2*np.pi*smooth((t-1.5)/3.5)
  xyz=np.array([20+8*np.cos(theta),216+9*np.sin(theta),0.]); down=1
 elif t<=6.5:
  f=smooth((t-5)/1.5); xyz=np.array([28-5*f*f,216-17*f,0.]); down=1
 else: xyz=np.array([23.,199.,30*smooth((t-6.5)/1.5)]); down=0
 target=origin+R@xyz
 for _ in range(80):
  err=target-fk(q)
  if np.linalg.norm(err)<.01: break
  jac=np.column_stack([(fk(q+np.eye(5)[j]*.01)-fk(q-np.eye(5)[j]*.01))/.02 for j in range(5)])
  dq=jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(3),err)
  q=np.clip(q+np.clip(dq,-2,2),LOW,HIGH)
 error=float(np.linalg.norm(target-fk(q))); errors.append(error)
 rows.append([float(t),*q,float(first['gripper.pos']),*xyz,down,error])
if max(errors)>.1: raise RuntimeError(max(errors))
with (out/'physical_9_PLAYBACK.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*[n+'.pos' for n in JOINTS],'gripper.pos','paper_x_mm','paper_y_mm','paper_z_mm','pen_down','fk_residual_mm']);w.writerows(rows)
vel=np.diff(np.array(rows)[:,1:6],axis=0)*30
report={'text':'9','duration_s':8.,'max_model_ik_error_mm':max(errors),'max_joint_speed_deg_s':float(np.abs(vel).max()),'paper_contact_z_mm':0,'lift_mm':30,'physical_accuracy_validated':False,'paper_frame':'same replay-aligned estimate as 21','motion':'quintic timing for lowering, oval, tail, and lift'}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
