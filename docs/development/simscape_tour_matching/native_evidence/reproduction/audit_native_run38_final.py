"""Read-only experiment: independently replay selected saved evaluations."""
import hashlib, json
from pathlib import Path
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate, replay_window
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
root=Path('/mnt/c/Users/diete'); run=root/'native-ms-pose-seed-fit-9967-38'
out=root/'native-run38-final-audit.json'
assert not out.exists()
raw=(root/'native_geometry_spec_9967.json').read_bytes(); spec=json.loads(raw); mh=hashlib.sha256(raw).hexdigest()
payload=json.loads((root/'driver_marker_payload_9967.json').read_text()); config=json.loads((run/'config.json').read_text())
assert mh==config['model_sha256']
times=np.array(payload['time_s']); times=times[times<=config['horizon']]
ends=config['nodes']; scales=np.array(config['defect_scales']); rows=[]
for number in [13]:
    path=run/f'evaluation-{number:05d}.json'; snapshot=json.loads(path.read_text())
    assert snapshot['config_sha256']==hashlib.sha256((run/'config.json').read_bytes()).hexdigest()
    c=NativeReplayCandidate.from_document(snapshot['candidate'],spec['coordinate_order'],mh)
    assert c.sha256==snapshot['candidate_sha256'] and c.document['capture_sha256']==payload['source_sha256']
    ids=[payload['labels'].index(label) for label in c.document['marker_labels']]
    target=np.array(payload['points_world_m'])[:len(times),ids]; valid=np.array(payload['valid'],bool)[:len(times),ids]&np.isfinite(target).all(axis=2)
    states={float(k):np.array(v) for k,v in snapshot['physical_nodes'].items()}
    full=replay_candidate(raw,c,times,rtol=1e-11,atol=1e-13,max_step=.00025)
    defects={}; start=0.
    for end in ends:
        state=np.array(c.document['q0']+c.document['qd0']) if start==0 else states[start]
        clock=times[(times>=start)&(times<=end)]
        result=replay_window(raw,c,clock,state,rtol=1e-11,atol=1e-13,max_step=.00025)
        if end in states: defects[str(end)]=float(np.linalg.norm((result.integration.state[-1]-states[end])/scales))
        start=end
    def rms(a,b,mask): return float(np.sqrt(np.mean(np.sum((a[mask]-b[mask])**2,axis=-1))))
    rows.append({'evaluation':number,'candidate_sha256':c.sha256,'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'callback_cost':snapshot['residual_sum_squares'],'whole_rms_m':rms(full.markers_m,target,valid),'terminal_rms_m':rms(full.markers_m[-1],target[-1],valid[-1]),'terminal_segmented_rms_m':rms(result.markers_m[-1],target[-1],valid[-1]),'terminal_pointwise_gap_m':rms(result.markers_m[-1],full.markers_m[-1],valid[-1]),'defect_norms':defects,'max_scaled_defect':max(defects.values())})
with out.open('x') as stream: json.dump({'qualification':'Selected residual evaluations, not accepted optimizer iterates; independent primal replay, no fit modification','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows},stream,indent=2)
