"""Offline conservative joint smoothing; no hardware imports or execution."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[json.loads(line) for line in (HERE/'teleop_demo_20261011T182444Z.jsonl').read_text().splitlines()]
keys=list(rows[0]['follower_observation'])
keys=[k for k in keys if k.endswith('.pos')]
weights=[1,4,6,4,1]
with (HERE/'smiley_smoothed_joint_candidate.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_s',*keys])
 for i,row in enumerate(rows):
  values=[]
  for k in keys:
   value=row['follower_observation'][k]
   if k!='gripper.pos' and 2<=i<len(rows)-2:
    value=sum(weight*rows[i+j-2]['follower_observation'][k] for j,weight in enumerate(weights))/16
   values.append(value)
  w.writerow([row['elapsed_s']-rows[0]['elapsed_s'],*values])
(HERE/'status.json').write_text(json.dumps({'samples':len(rows),'duration_s':rows[-1]['elapsed_s']-rows[0]['elapsed_s'],'method':'Five-sample binomial smoothing of recorded follower joints; gripper unchanged','physically_executed':False,'limitations':['Conservative jitter smoothing only, not fitted face geometry.','Recording includes approach and idle periods.','Pen contact labels and stroke boundaries have not yet been determined.','Not prepared for direct physical replay.']},indent=2)+'\n')
print('Saved recording and offline smoothed candidate:',len(rows),'samples')
