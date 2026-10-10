from pathlib import Path
import hashlib,json,re,sys
from decimal import Decimal
ROOT=Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift');P=ROOT/'docs/research/shot_pattern_analysis/phase2';OUT=Path('/tmp/p2-article-astra-review');sys.path.insert(0,str(ROOT))
from scripts.research.shot_pattern_analysis.final_report_inputs import select_representative_interaction_pairs,select_bounds_control_pairs
from scripts.research.shot_pattern_analysis.report_synthesis import validate_resolved_selection

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
trace_path=P/'validation/selected_headline_traceability_v1/review.json';trace=json.loads(trace_path.read_text())
for name,digest in trace['source_sha256'].items():
 if sha(P/name)!=digest:raise ValueError(name)
cells=[]
for entry in trace['cells']:
 s=json.loads((P/entry['source_root']/'summary.json').read_text());cells.append({**s,'cell':entry['cell'],'source_protocol':entry['source_protocol'],'source_root':entry['source_root']})
validate_resolved_selection(cells)
expected=json.loads((P/'validation/protocol_name_recovery_v6/expected_numeric_hashes.json').read_text());inputs={str(trace_path):sha(trace_path)}
for name,digest in expected.items():
 p=P/'supplement_report_v1'/name
 if sha(p)!=digest:raise ValueError(name)
 inputs[str(p)]=digest
pairs=json.loads((P/'supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json').read_text())['comparisons'];effects=select_representative_interaction_pairs(pairs);controls,missing=select_bounds_control_pairs(pairs)
if missing or len(effects)!=12 or len(controls)!=2:raise ValueError('selector inventory')
article=(OUT/'rejected_article.tex').read_text();rows=[];failures=[]
def check(label,token,value):
 d=Decimal(token);tol=float(Decimal(10)**d.as_tuple().exponent)/2+abs(float(value))*1e-12
 ok=abs(float(token)-value)<=tol
 result={'field':label,'display':token,'expected':value,'rounding_tolerance':tol,'matches_display_precision':ok};rows.append(result)
 if not ok:failures.append(result)
count=0
for line in article.splitlines():
 if not re.match(r'C\d\d\s*&\s*(Straight|Draw|Fade)\s*&',line):continue
 fields=[v.strip() for v in line.rstrip('\\ ').split('&')];i=int(fields[0][1:])-1;pattern=fields[1];cell=cells[i];stats=cell['summary'][pattern];score=cell['course_scoring'];count+=1
 ns=[int(v.replace(',','').strip()) for v in fields[2].split('/')]
 for label,actual,want in [('support',ns[0],stats['supported_n']),('attempts',ns[1],stats['total_n'])]:
  check(f'{fields[0]} {pattern} {label}',str(actual),want)
 for column,key in [(3,'lateral_sd_m'),(4,'downrange_sd_m'),(5,'target_rmse_m')]:check(f'{fields[0]} {pattern} {key}',fields[column],stats[key])
 check(f'{fields[0]} {pattern} absolute_sg',fields[6],score['patterns'][pattern]['mean_strokes_gained'])
 if pattern!='Straight':
  numbers=re.findall(r'[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?',fields[7]);interval=score['paired_benefit_vs_straight'][pattern]
  for token,key in zip(numbers,('estimate','lower_95','upper_95'),strict=True):check(f'{fields[0]} {pattern} {key}',token,interval[key])
if count!=72:raise ValueError(count)
# Exact selected control records and source summaries, including signed face/path.
control_records=[]
for effect in controls:
 match=next(r for r in pairs if r['comparison_kind']=='across_lean' and [(s['protocol_id'],s['cell']) for s in r['sources']]==list(zip(effect['source_protocols'],effect['source_cells'])))
 control_records.append(match)
nominals=[]
report=json.loads((P/'supplement_report_v1/REPORT_SUPPLEMENT.json').read_text());supp={ (r['source_protocol'],r['cell']):r for r in report['unique_cells']}
for pair in control_records:
 for src in pair['sources']:
  key=(src['protocol_id'],src['cell'])
  if key in supp: ref=supp[key];summary=ref['summary'];path=Path(ref['source_root'])/ref['cell']/'summary.json'
  else:path=P/'control_final_v1'/src['cell']/'summary.json';summary=json.loads(path.read_text())
  inputs[str(path)]=sha(path);nominals.append({'protocol':src['protocol_id'],'cell':src['cell'],'config':summary['config'],'target':summary['target_x_m'],'nominals':summary['nominals'],'unsupported':{k:summary['summary'][k]['unsupported_reasons'] for k in ['Straight','Draw','Fade']}})
source_paths=['scripts/research/shot_pattern_analysis/final_report_inputs.py','scripts/research/shot_pattern_analysis/report_synthesis.py','src/tools/shot_pattern_analysis/matched_interpretation.py','src/tools/shot_pattern_analysis/matched_formatting.py']
data={'cells':cells,'effects':effects,'controls':controls,'control_records':control_records,'nominals':nominals,'input_sha256':inputs,'api_sha256':{p:sha(ROOT/p) for p in source_paths}}
(OUT/'verified_data.json').write_text(json.dumps(data,indent=2)+'\n')
review={'candidate_sha256':sha(OUT/'rejected_article.tex'),'status':'changes_required','checked_pattern_rows':count,'checked_numeric_fields':len(rows),'numeric_rounding_mismatches':failures,'all_field_checks':rows,'max_selected_nominal_residual_m':max(n['residual_m'] for c in cells for n in c['nominals'].values()),'course_contexts':[{k:c['course_scoring'].get(k) for k in ['club_id','start_lie','start_distance_m','hole_distance_m','fairway_half_width_m','green_radius_m']} for c in cells[::8]],'selected_control_identities':[{k:e[k] for k in ['source_protocols','source_cells','common_six_n']} for e in controls],'sources':data['input_sha256']}
(OUT/'candidate_field_audit.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({k:review[k] for k in ['checked_pattern_rows','checked_numeric_fields','numeric_rounding_mismatches','max_selected_nominal_residual_m','course_contexts','selected_control_identities']},indent=2))
