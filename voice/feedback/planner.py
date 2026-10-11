"""Bounded paper-space changes using the existing calibrated pen-tip IK."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'roboalgo'))
from prepare_physical_21 import fk,JOINTS,LOW,HIGH

def build(base,folder,changes):
 source=Path(base['directory']);jointfile,paperfile,runner=base['files']
 rows=list(csv.DictReader((source/paperfile).read_text().splitlines()));joints=list(csv.DictReader((source/jointfile).read_text().splitlines()))
 t=np.array([float(r['time_s']) for r in rows]);xyz=np.array([[float(r['paper_'+a+'_mm']) for a in 'xyz'] for r in rows]);down=np.array([int(float(r['pen_down'])) for r in rows],bool)
 if base['routine']=='9':
  cfg=json.loads((ROOT/'roboalgo/examples/9_physical/smoothed_demo/report.json').read_text());surface=np.array(cfg['surface_z_from_xy_coefficients']);nominal_lift=20
  tail=t>10;start=xyz[np.argmin(abs(t-10)),:2].copy();factor=changes.get('stem_scale',1);xyz[tail,:2]=start+factor*(xyz[tail,:2]-start)
 else:
  cfg=json.loads((source/'report.json').read_text());surface=np.array(cfg['inferred_contact_surface']);nominal_lift=40
  starts=np.flatnonzero(down & ~np.r_[False,down[:-1]]);assert len(starts)==3
  mouth=starts[-1];center=(xyz[starts[0],0]+xyz[starts[1],0])/2
  # Blend eye-spacing displacement to zero before mouth descent.
  mouth_start_x=xyz[mouth,0];spacing=changes.get('eye_spacing_scale',1)
  shift=np.zeros(len(t));shift[:starts[1]+1]=(spacing-1)*(xyz[:starts[1]+1,0]-center)
  end=max(starts[1]+1,mouth-75);shift[starts[1]:end+1]=np.linspace(shift[starts[1]],0,end-starts[1]+1)
  xyz[:,0]+=shift
  top=xyz[mouth,1];xyz[mouth:,1]=top+(xyz[mouth:,1]-top)*changes.get('mouth_depth_scale',1)
 old=np.array([[float(r['paper_'+a+'_mm']) for a in 'xyz'] for r in rows]);lift=old[:,2]-(old[:,:2]@surface[:2]+surface[2])
 xyz[:,2]+=((xyz[:,:2]-old[:,:2])@surface[:2])+changes.get('lift_delta_mm',0)*np.clip(lift/nominal_lift,0,1)
 s=json.loads((ROOT/'roboalgo/examples/21/source_sim/summary.json').read_text());R=np.array(s['paper_to_world_rotation']);origin=np.array(s['paper_origin_world_m'])*1000
 if np.any(xyz[:,0]<0) or np.any(xyz[:,0]>170) or np.any(xyz[:,1]<0) or np.any(xyz[:,1]>235):raise ValueError('Path leaves calibrated drawing rectangle')
 q=np.array([float(joints[0][j+'.pos']) for j in JOINTS]);out=[];errors=[]
 for point in xyz:
  target=origin+R@point
  for _ in range(120):
   err=target-fk(q)
   if np.linalg.norm(err)<.02:break
   jac=np.column_stack([(fk(q+np.eye(5)[k]*.01)-fk(q-np.eye(5)[k]*.01))/.02 for k in range(5)])
   q=np.clip(q+np.clip(jac.T@np.linalg.solve(jac@jac.T+.01*np.eye(3),err),-2,2),LOW,HIGH)
  out.append(q.copy());errors.append(np.linalg.norm(target-fk(q)))
 out=np.array(out);speed=float(np.max(abs(np.diff(out,axis=0))/np.diff(t)[:,None]))
 if max(errors)>.1 or speed>30:raise ValueError('Candidate rejected: IK residual or joint speed limit')
 folder.mkdir(parents=True)
 with (folder/jointfile).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_s',*[j+'.pos' for j in JOINTS],'gripper.pos','pen_down']);w.writerows([tt,*q,float(j['gripper.pos']),int(d)] for tt,q,j,d in zip(t,out,joints,down))
 with (folder/paperfile).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['time_s','paper_x_mm','paper_y_mm','paper_z_mm','pen_down']);w.writerows([tt,*p,int(d)] for tt,p,d in zip(t,xyz,down))
 code=(source/runner).read_text();code=re.sub(r" if len\(plan\).*?raise RuntimeError\('Unexpected trajectory'\)",f" if len(plan)!={len(t)} or abs(duration-{t[-1]})>1e-5:raise RuntimeError('Unexpected trajectory')",code)
 compile(code,str(folder/runner),'exec');(folder/runner).write_text(code)
 report={'max_ik_residual_mm':float(max(errors)),'max_joint_speed_deg_s':speed,'physically_validated':False,'changes':changes}
 (folder/'report.json').write_text(json.dumps(report,indent=2));return report
