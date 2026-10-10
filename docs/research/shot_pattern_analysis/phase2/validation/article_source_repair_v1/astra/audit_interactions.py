from pathlib import Path
import json,re
from decimal import Decimal
p=Path('/tmp/p2-article-astra-review');d=json.loads((p/'verified_data.json').read_text());s=(p/'rejected_article.tex').read_text();key=None;checks=[]
for line in s.splitlines():
 f=[x.strip() for x in line.rstrip('\\ ').split('&')]
 if re.match(r'E\d\d &',line):key=int(f[0][1:])-1
 elif not(key is not None and len(f)==6 and f[3]=='Fade'):continue
 if len(f)!=6 or f[3] not in ['Draw','Fade']:continue
 e=d['effects'][key];v=e[f[3].lower()+'_minus_straight_change'];n=int(f[4].replace(',',''));tokens=re.findall(r'[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?',f[5])
 if n!=e['common_six_n']:raise ValueError('population')
 for t,k in zip(tokens,['estimate','lower_95','upper_95'],strict=True):
  tolerance=float(Decimal(10)**Decimal(t).as_tuple().exponent)/2+1e-12
  checks.append({'key':f'E{key+1:02}','pattern':f[3],'field':k,'display':t,'actual':v[k],'matches':abs(float(t)-v[k])<=tolerance})
 if f[3]=='Fade':key=None
if len(checks)!=72:raise ValueError(len(checks))
(p/'candidate_interaction_audit.json').write_text(json.dumps({'checked_fields':len(checks),'failures':[c for c in checks if not c['matches']],'checks':checks},indent=2)+'\n')
print('72 interaction fields checked; failures:',sum(not c['matches'] for c in checks))
