from pathlib import Path
import re
p=Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/shot_pattern_analysis.tex');s=p.read_text()
s=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',lambda m:re.sub(r'\n\s*\n','\n',m[0]),s,flags=re.S)
s=s.replace(r'\date{October 8, 2026}',r'\date{October 10, 2026}')
s=s.replace(r'\texttt{table9-full-plus-anchor-reconciled-putting/2}',r'\path{table9-full-plus-anchor-reconciled-putting/2}')
for name in ['driver_lean0_curve2_rotation2_combined2','driver_lean5_curve2_rotation2_combined2','pitching_wedge_lean0_curve2_rotation2_combined2','pitching_wedge_lean5_curve2_rotation2_combined2']:
 s=s.replace(r'\texttt{'+name.replace('_',r'\_')+'}',r'\path{'+name+'}')
s=s.replace('SG units are strokes per modeled opportunity.','SG units are strokes per modeled opportunity. Numerical notation such as $2.5$e$-5$ means $2.5\\times10^{-5}$; interval endpoints retain seven significant digits. Display rounding does not add model accuracy.')
p.write_text(s);Path('/tmp/p2-article-astra-review/repaired_article.tex').write_text(s)
print(len(s.splitlines()))
