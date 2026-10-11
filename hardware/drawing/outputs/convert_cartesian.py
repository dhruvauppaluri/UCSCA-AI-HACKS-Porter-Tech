"""Convert recorded joint samples to estimated follower-base pen XYZ (mm)."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from fk_so101 import tip_position, URDF, TIP_OFFSET

BASE = Path(__file__).resolve().parents[1]
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')

def convert(source, destination):
    if source.suffix == '.csv':
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        groups = [('leader_model', '')] if 'shoulder_pan.pos' in rows[0] else [
            ('leader_model', 'leader_'), ('follower', 'follower_'), ('target', 'target_')]
        samples = [(row['time_s'], {label: {j: float(row[prefix+j+'.pos']) for j in JOINTS}
                    for label, prefix in groups}) for row in rows]
    else:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        groups = [('leader_model', 'leader_action'), ('follower', 'follower_observation'), ('target', 'sent_action')]
        samples = [(row['elapsed_s'], {label: {j: float(row[key][j+'.pos']) for j in JOINTS}
                    for label, key in groups}) for row in rows]
    fields = ['source_row', 'segment_index', 'time_s'] + [f'{label}_{axis}_mm' for label, _ in groups for axis in 'xyz']
    previous = None
    segment = 0
    values = []
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (time_s, joints) in enumerate(samples, 1):
            t = float(time_s)
            if previous is not None and t < previous:
                segment += 1
            previous = t
            result = {'source_row': index, 'segment_index': segment, 'time_s': time_s}
            for label, q in joints.items():
                xyz = tip_position(q) * 1000
                if not np.isfinite(xyz).all():
                    raise ValueError(f'Nonfinite position: {source}:{index}')
                result.update({f'{label}_{axis}_mm': round(float(value), 6) for axis, value in zip('xyz', xyz)})
            writer.writerow(result)
            values.append(result)
    with destination.open() as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    return {'source': str(source.relative_to(BASE)), 'output': destination.name,
            'rows': len(rows), 'segments': segment+1,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}

def main():
    output = BASE/'cartesian'
    output.mkdir(exist_ok=True)
    sources = sorted((BASE/'robot-host').rglob('*.csv'))
    sources += sorted((BASE/'remote-data').glob('teleop_demo_*.jsonl'))
    sources += sorted((BASE/'remote-data').glob('three_circles_*.jsonl'))
    manifest = {'frame': 'URDF base_link; meters converted to millimeters',
                'urdf': str(URDF.relative_to(BASE)),
                'urdf_sha256': hashlib.sha256(URDF.read_bytes()).hexdigest(),
                'tip_offset_gripper_frame_mm': (TIP_OFFSET*1000).tolist(),
                'files': [convert(p, output/(p.stem+'_cartesian.csv')) for p in sources]}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
