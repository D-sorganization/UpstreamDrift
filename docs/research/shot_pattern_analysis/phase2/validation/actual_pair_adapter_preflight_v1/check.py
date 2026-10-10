from pathlib import Path
import json,sys,hashlib,collections
ROOT=Path('/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift')
sys.path.insert(0,str(ROOT))
from scripts.research.shot_pattern_analysis.supplement_revision_adapter import reconstruct_pair
from scripts.research.shot_pattern_analysis.supplement_publication import sha256
P=Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2')
OUT=Path('/tmp/shot-pattern-actual-pair-adapter-review')
files=[P/'supplement_report_v1'/n for n in ['REPORT_SUPPLEMENT.json','REPORT_SUPPLEMENT_COMPARISONS.json']]
before={str(p):sha256(p) for p in files}
report,comparisons=[json.loads(p.read_text()) for p in files]
roots={'original_primary/1':P/'final_v1','original_control/1':P/'control_final_v1','pw-primary-face-bounds/1':P/'supplement_final_v1/pw_primary_face','pw-control-face-bounds/1':P/'supplement_final_v1/pw_control_face','driver-control-speed-bounds/1':P/'supplement_final_v1/driver_control_speed'}
snapshots={}; manifests={};rows=[];failures=[]
for row in comparisons['comparisons']:
 for proof in row['sources']:
  key=f"{proof['protocol_id']}:{proof['cell']}"
  if key in snapshots:continue
  root=roots[proof['protocol_id']];directory=root/proof['cell']; mp=directory/'manifest.json';m=json.loads(mp.read_text());s=json.loads((directory/'summary.json').read_text())
  assert sha256(mp)==proof['source_cell_manifest_sha256']
  assert sha256(root/'matrix_plan.json')==proof['source_plan_sha256']
  for name,digest in m['files_sha256'].items():assert sha256(directory/name)==digest,(key,name)
  assert m['files_sha256']['shots.csv.gz']==proof['source_csv_sha256']
  manifests[str(mp)]=sha256(mp)
  snapshots[key]={'source_root':str(root),'status':s['status'],'config':s['config'],'manifest_sha256':sha256(mp),'summary_sha256':m['files_sha256']['summary.json'],'csv_sha256':m['files_sha256']['shots.csv.gz']}
loaded={'report':report,'comparisons':comparisons,'receipt':{'source_cells_sha256':snapshots}}
for row in comparisons['comparisons']:
 a,b=row['sources']; rec={'comparison_kind':row['comparison_kind'],'first_protocol':a['protocol_id'],'first_cell':a.get('alias_cell') or a['cell'],'second_protocol':b['protocol_id'],'second_cell':b.get('alias_cell') or b['cell'],'target_x_m':row['target_x_m'],'source_kind':'six_population_common_index_comparison','common_supported_n':row['common_supported_n'],'total_n_per_population':row['total_n_per_population']}
 try:
  pair=reconstruct_pair(loaded,rec)
  assert pair['first']['source_protocol']==a['protocol_id'] and pair['second']['source_protocol']==b['protocol_id']
  rows.append({**rec,'alias_count':sum(x.get('execution_kind')=='provenance_alias' for x in [pair['first'],pair['second']]),'status':'reconstructed'})
 except Exception as e:failures.append({**rec,'error':repr(e)})
assert before=={str(p):sha256(p) for p in files}
assert all(sha256(Path(p))==h for p,h in manifests.items())
result={'scope':'Provisional Actual Ordered-Pair Adapter Preflight','parent_final_acceptance':False,'parent_receipt_available':(P/'supplement_report_v1/REPORT_SUPPLEMENT_PIPELINE.json').exists(),'input_sha256':before,'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'scripts/research/shot_pattern_analysis/supplement_revision_adapter.py',ROOT/'scripts/research/shot_pattern_analysis/supplement_publication.py']},'unique_bound_cells':len(snapshots),'comparisons':len(comparisons['comparisons']),'reconstructed':len(rows),'counts':dict(collections.Counter(r['comparison_kind'] for r in rows)),'alias_references':sum(r['alias_count'] for r in rows),'failures':failures,'all_source_manifests_sha256':manifests,'rows':rows,'no_statistics_or_rendering_executed':True,'script_sha256':sha256(Path(__file__))}
(OUT/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['comparisons','reconstructed','unique_bound_cells','counts','alias_references','failures']},indent=2))
