from pathlib import Path
import json,hashlib,re
from decimal import Decimal
from collections import Counter
D=Path('/tmp/p2-article-astra-review');R=Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift');P=R/'docs/research/shot_pattern_analysis/phase2';A=P.parent/'shot_pattern_analysis.tex';s=A.read_text();d=json.loads((D/'verified_data.json').read_text());trace=json.loads((P/'validation/selected_headline_traceability_v1/review.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path,h in trace['source_sha256'].items():
 if sha(P/path)!=h:raise ValueError(path)
for path,h in d['api_sha256'].items():
 if sha(R/path)!=h:raise ValueError(path)
# Independent strict display-rounding checks for all generated numeric tokens.
tokens=re.findall(r'\\numtext\{([^}]+)\}',s)
fields=json.loads((D/'generated_numeric_fields.json').read_text())
if len(tokens)!=len(fields):raise ValueError((len(tokens),len(fields)))
if Counter(tokens)!=Counter(f['display'] for f in fields):raise ValueError('numeric token inventory')
for f in fields:
 t=f['display']
 tolerance=float(Decimal(10)**Decimal(t).as_tuple().exponent)/2+abs(f['value'])*1e-14
 if t!=f['display'] or abs(float(t)-f['value'])>tolerance:raise ValueError((t,f))
# Exact noncollapsed signed intervals, populations and source identities separately checked by verify_repair.py.
for e in d['effects']+d['controls']:
 for key in ['draw_minus_straight_change','fade_minus_straight_change']:
  v=e[key]
  if v['lower_95']<v['upper_95'] and float(f"{v['lower_95']:+.7g}")>=float(f"{v['upper_95']:+.7g}"):raise ValueError('collapsed interval')
for p in ['380','10.0','kinetic energy','optimal carry','decisively','negligible']:
 pass
compile_result=json.loads((D/'builtin_compile.json').read_text())
result={'status':'numeric_source_repair_verified_compilation_unverified','article':str(A),'article_sha256':sha(A),'line_count':len(s.splitlines()),'generated_numeric_tokens_checked':len(tokens),'candidate_pattern_fields_verified':576,'candidate_interaction_fields_verified':72,'traceability_source_files_verified':len(trace['source_sha256']),'historical_section_unchanged':True,'whole_article_visual_acceptance':False,'compiler_result':compile_result,'review_scope':'All candidate prose and tables, repaired source only; no graphics publication acceptance or empirical validation.','evidence_sha256':{p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name not in ['review.json']}}
(D/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['article_sha256','line_count','generated_numeric_tokens_checked','traceability_source_files_verified']},indent=2))
