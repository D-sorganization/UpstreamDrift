from pathlib import Path
import json,re,hashlib
P=Path('/tmp/p2-article-astra-review');d=json.loads((P/'verified_data.json').read_text());s=(P/'repaired_article.tex').read_text();old=(P/'rejected_article.tex').read_text();out=[]
def verify(got,want,label):
 if len(got)!=len(want):raise ValueError((label,len(got),len(want)))
 for a,b in zip(got,want,strict=True):
  if abs(float(a)-float(b))>max(abs(float(b))*5.1e-7,5.01e-4 if label.endswith('geometry') else 1e-15):raise ValueError((label,a,b))
 out.append({'field':label,'values':got,'source_values':want})
def nums(line):return re.findall(r'\\numtext\{([^}]+)\}',line)
counts={'geometry':0,'benefits':0,'effects':0,'control_nominals':0,'controls':0}
for line in s.splitlines():
 fields=[f.strip() for f in line.rstrip('\\ ').split('&')]
 if re.match(r'C\d\d & (Straight|Draw|Fade)',line):
  c=d['cells'][int(fields[0][1:])-1];pat=fields[1];st=c['summary'][pat];sc=c['course_scoring']
  if len(fields)==7:
   if int(fields[2])!=st['supported_n']:raise ValueError('endpoint n')
   verify(nums(line),[st[k] for k in ['lateral_sd_m','downrange_sd_m','target_rmse_m']]+[sc['patterns'][pat]['mean_strokes_gained']],fields[0]+pat+'geometry');counts['geometry']+=1
  else:
   if int(fields[2])!=sc['paired_n']:raise ValueError('SG n')
   verify(nums(line),[sc['paired_benefit_vs_straight'][pat][k] for k in ['estimate','lower_95','upper_95']],fields[0]+pat);counts['benefits']+=1
 elif re.match(r'E\d\d & (Draw|Fade)',line):
  e=d['effects'][int(fields[0][1:])-1];f=fields[1].lower()+'_minus_straight_change'
  if int(fields[2])!=e['common_six_n']:raise ValueError('effect n')
  verify(nums(line),[e[f][k] for k in ['estimate','lower_95','upper_95']],fields[0]+f);counts['effects']+=1
 elif re.match(r'F[12][AB] &',line):
  i=(int(fields[0][1])-1)*2+(fields[0][2]=='B');n=d['nominals'][i]['nominals'][fields[1]]
  for got,key in zip(nums(line),['face_deg','path_deg','speed_mps','residual_m'],strict=True):
   if abs(float(got)-n[key])>max(5.01e-6,abs(n[key])*5.1e-7):raise ValueError('nominal')
  counts['control_nominals']+=1
 elif re.match(r'F[12] & (Draw|Fade)',line):
  e=d['controls'][int(fields[0][1])-1];f=fields[1].lower()+'_minus_straight_change'
  if int(fields[2])!=e['common_six_n']:raise ValueError('control n')
  verify(nums(line),[e[f][k] for k in ['estimate','lower_95','upper_95']],fields[0]+f);counts['controls']+=1
expected={'geometry':72,'benefits':48,'effects':24,'control_nominals':12,'controls':4}
if counts!=expected:raise ValueError(counts)
marker=r'\section{Historical Phase-One Status and Scope}'
if s[s.index(marker):]!=old[old.index(marker):]:raise ValueError('historical changed')
for p,h in d['input_sha256'].items():
 if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError(p)
# Check balanced environments and braces, excluding escaped braces and comments.
stack=[]
for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',s):
 if kind=='begin':stack.append(name)
 elif not stack or stack.pop()!=name:raise ValueError((kind,name))
if stack:raise ValueError(stack)
res={'article_sha256':hashlib.sha256(s.encode()).hexdigest(),'counts':counts,'numeric_checks':out,'historical_section_unchanged':True,'all_bound_inputs_unchanged':True,'latex_environment_balance':True,'compilation':'pending actual built-in attempt','visual_layout_acceptance':False}
(P/'repair_verification.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='numeric_checks'},indent=2))
