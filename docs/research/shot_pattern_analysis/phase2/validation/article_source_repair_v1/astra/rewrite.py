from pathlib import Path
import json,hashlib
P=Path('/tmp/p2-article-astra-review');d=json.loads((P/'verified_data.json').read_text());cells=d['cells'];effects=d['effects'];controls=d['controls']
article=Path('/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/shot_pattern_analysis.tex')
s=article.read_text(); start=s.index(r'\subsection{Selected Zero-Lean Experimental Design and Condition Keys}');end=s.index(r'\clearpage'+'\n'+r'\section{Historical Phase-One Status and Scope}',start)
lines=[]; checks=[]
def add(x=''):lines.append(x)
def num(v,places=None):
 t=f'{v:.{places}f}' if places is not None else f'{v:+.7g}'
 checks.append({'display':t,'value':v})
 return r'\numtext{'+t+'}'
def table(title,headers,rows,spec=None):
 add(r'\begin{longtable}{'+(spec or 'l'*len(headers))+'}')
 add(r'\caption{'+title+r'}\\')
 add(r'\toprule '+ ' & '.join(headers)+r' \\ \midrule')
 add(r'\endfirsthead');add(r'\toprule '+' & '.join(headers)+r' \\ \midrule');add(r'\endhead')
 for row in rows:add(' & '.join(map(str,row))+r' \\')
 add(r'\bottomrule\end{longtable}')
protocol={'original_primary/1':'P','original_control/1':'OC','pw-primary-face-bounds/1':'WP','pw-control-face-bounds/1':'WC','driver-control-speed-bounds/1':'DC'}
keys={c['cell']:f'C{i+1:02}' for i,c in enumerate(cells)}
clubnames={'driver':'Driver','seven_iron':'7-Iron','pitching_wedge':'Pitching Wedge'}
add(r'\subsection{Selected Zero-Lean Conditions and Exact Source Identities}')
add('The selected analysis contains 24 zero-lean conditions and 72 pattern rows: three clubs, two curve scales, two shaft-twist SDs, and two noise profiles. Each pattern attempts 10,000 shared-seed deliveries. These are selected strata of the full matrix, not independent player trials.')
add(r'Protocol keys identify distinct source records: P = \texttt{original\_primary/1}; OC = \texttt{original\_control/1}; WP = \texttt{pw-primary-face-bounds/1}; WC = \texttt{pw-control-face-bounds/1}; DC = \texttt{driver-control-speed-bounds/1}.')
add(r'The four PW curve-$2$ headline conditions use WP. Original bounded-search unavailable records remain preserved. Original matching searches reference face $\pm3^\circ$; WP and WC use absolute face bounds $[-6,6]^\circ$ with speed factors $[0.75,1.25]$. These supplemental bounds were declared after feasibility review, before supplemental sampling; they are not the original protocol.')
add(r'Keys identify the cell names \texttt{\detokenize{<club>_lean0_curve<C>_rotation<SD>_<profile>}}. Club identifiers are \texttt{driver}, \texttt{seven\_iron}, and \texttt{pitching\_wedge}. S denotes \texttt{shaft\_only}; C denotes \texttt{combined2}. The profile C adds path SD $0.5^\circ$, speed SD $2$ mph ($0.89408$ m/s), and horizontal and vertical strike SD $0.25$ in ($0.00635$ m) each, with independent primitive errors. S holds the other primitive errors at zero. Neither profile makes delivered face SD equal to shaft-twist SD.')
rows=[]
for i,c in enumerate(cells):
 cfg=c['config'];rows.append([f'C{i+1:02}',clubnames[cfg['club']['club_id']],int(cfg['curve_scale']),int(cfg['noise']['standard_deviations'][0]),'S' if c['cell'].endswith('shaft_only') else 'C',protocol[c['source_protocol']]])
add(r'{\small');table('Selected Condition and Protocol Keys',['Key','Club','Curve','Twist SD ($^\circ$)','Profile','Protocol'],rows);add('}')
add('Targets are Driver 178.948758 m, 7-Iron 127.881942 m, and PW 77.402477 m. Face, path and speed are solved before sampling, with no post-flight aim correction. The maximum residual among the 72 selected nominal solutions is '+num(max(n['residual_m'] for c in cells for n in c['nominals'].values()))+r' m, below the predeclared $0.05$ m criterion. Matching nominal endpoints does not constrain sampled mean carry or remove longitudinal error.')
add(r'\subsection{Dispersion, Absolute Scoring, and Within-Condition Benefits}')
add(r'All geometry columns below use each pattern\textquotesingle s individual supported endpoints. $n_e$ is that endpoint count; every row attempts 10,000. Absolute SG and relative SG use the common-three scored population $n_3$ shown in the following tables, which can differ from individual support. Unsupported cases remain in accounting and are not resampled. RMSE is radial target error; $\sigma_Y$ and $\sigma_X$ are lateral and downrange SDs, all in metres. SG units are strokes per modeled opportunity.')
for club in clubnames:
 subset=[c for c in cells if c['config']['club']['club_id']==club];rows=[];benefits=[]
 for c in subset:
  for pat in ['Straight','Draw','Fade']:
   st=c['summary'][pat];sc=c['course_scoring'];rows.append([keys[c['cell']],pat,st['supported_n'],num(st['lateral_sd_m'],3),num(st['downrange_sd_m'],3),num(st['target_rmse_m'],3),num(sc['patterns'][pat]['mean_strokes_gained'])])
   if pat!='Straight':
    v=sc['paired_benefit_vs_straight'][pat];benefits.append([keys[c['cell']],pat,sc['paired_n'],*[num(v[k]) for k in ['estimate','lower_95','upper_95']]])
 add(r'{\small');table(clubnames[club]+' Endpoint Geometry and Absolute SG',['Key','Pattern',r'$n_e$',r'$\sigma_Y$',r'$\sigma_X$','RMSE','Mean SG'],rows);table(clubnames[club]+' Paired Benefit Relative to Straight',['Key','Pattern',r'$n_3$','Estimate','Lower 95\%','Upper 95\%'],benefits);add('}')
add(r'The scoring context is a 400 m tee opportunity for Driver, with a 15 m fairway half-width and rough outside. The 7-Iron and PW use fairway approach opportunities at their respective nominal target distances, with a centered circular green of radius 15 m and rough outside. First ground contact is treated as the stopping lie: bounce, roll and hazards are not modeled. The baseline is the historical Broadie-derived approximation, version \texttt{table9-full-plus-anchor-reconciled-putting/2}; positive-distance unholed putting expectations are bounded below by one stroke. Absolute SG is benchmark-dependent and is not a predicted handicap or per-round gain.')
add(r'\subsection{Paired SD and Curve-Strength Interactions}')
add(r'For pattern $P$, each interaction is the rowwise difference $[(SG_P-SG_S)_B-(SG_P-SG_S)_A]$ on the common supported indices of all six populations. Its interval is computed directly from paired resampling, not by subtracting marginal confidence limits. The source pairs below use OC where indicated, even when their cell names also occur in the headline P matrix. A C-key in this table identifies the cell name only; the adjoining protocol code completes the identity.')
rows=[];erows=[]
for i,e in enumerate(effects):
 key=f'E{i+1:02}';kind='Twist SD 1 to 2' if e['comparison_kind']=='shaft_rotation_sd' else 'Curve 1 to 2';a,b=e['source_cells'];pa,pb=e['source_protocols'];rows.append([key,kind,keys[a]+'/'+protocol[pa],keys[b]+'/'+protocol[pb]])
 for pat,f in [('Draw','draw_minus_straight_change'),('Fade','fade_minus_straight_change')]:erows.append([key,pat,e['common_six_n'],*[num(e[f][k]) for k in ['estimate','lower_95','upper_95']]])
add(r'{\small');table('Ordered Interaction Source Keys',['Key','Contrast','A','B'],rows);table('Direct Common-Six Relative-Pattern Interactions',['Key','Pattern',r'$n_6$','Estimate','Lower 95\%','Upper 95\%'],erows);add('}')
add(r'In these selected controls, increasing twist SD from $1^\circ$ to $2^\circ$ increases Draw relative to Straight for Driver and PW in both profiles. The 7-Iron SD interactions span zero; that is not proof of equivalence or negligible effects. Increasing curve scale from 1 to 2 at twist SD $2^\circ$ increases Driver Draw, increases 7-Iron Fade, and increases PW Draw while decreasing PW Fade in both profiles. These are conditional comparisons of complete matched deliveries. They do not isolate curvature as a causal intervention.')
add(r'\subsection{Fixed-Target Across-Lean Controls}')
add(r'The two selected controls both compare lean $0^\circ$ to $5^\circ$, curve scale 2, twist SD $2^\circ$, and combined2 noise. F1 is Driver: OC/\texttt{driver\_lean0\_curve2\_rotation2\_combined2} to DC/\texttt{driver\_lean5\_curve2\_rotation2\_combined2}. F2 is PW: WP/\texttt{pitching\_wedge\_lean0\_curve2\_rotation2\_combined2} to WC/\texttt{pitching\_wedge\_lean5\_curve2\_rotation2\_combined2}. F2\textquotesingle s zero-lean data also supplies a verified fixed-target control alias; it is reused data, not another simulation.')
add(r'Driver F1 holds target 178.948758 m; PW F2 holds 77.402477 m. DC uses absolute face bounds $[-6,6]^\circ$ and speed factors $[0.75,1.30]$, whereas its OC reference uses the original reference-centered face search and $[0.75,1.25]$ speed factors. PW WP and WC both use absolute $[-6,6]^\circ$ and $[0.75,1.25]$. The Driver contrast therefore includes an explicitly qualified speed-bound exception, not merely a lean change.')
rows=[]
for i,n in enumerate(d['nominals']):
 for pat in ['Straight','Draw','Fade']:
  q=n['nominals'][pat];rows.append([f'F{i//2+1}'+('A' if i%2==0 else 'B'),pat,num(q['face_deg'],5),num(q['path_deg'],5),num(q['speed_mps'],5),num(q['residual_m'])])
add(r'{\small');table('Signed Nominal Inputs for Fixed-Target Controls',['Key','Pattern','Face ($^\circ$)','Path ($^\circ$)','Speed (m/s)','Residual (m)'],rows)
rows=[]
for i,e in enumerate(controls):
 for pat,f in [('Draw','draw_minus_straight_change'),('Fade','fade_minus_straight_change')]:rows.append([f'F{i+1}',pat,e['common_six_n'],*[num(e[f][k]) for k in ['estimate','lower_95','upper_95']]])
table('Fixed-Target Common-Six Relative-Pattern Interactions',['Key','Pattern',r'$n_6$','Estimate','Lower 95\%','Upper 95\%'],rows);add('}')
add(r'The Driver common-six intersection excludes 60 of the 10,000 shared indices; the PW intersection excludes one. These counts are union exclusions across six populations, not counts of one diagnosed failure mechanism. At zero twist the geometric loft is static loft minus lean: Driver changes from $10^\circ$ to $5^\circ$, PW from $46^\circ$ to $41^\circ$. The solved speeds increase for Driver and decrease for PW as shown above. This is a model result, not evidence that increased kinetic energy caused unsupported shots or that an aerodynamic optimum was identified. Nominal face and path also change. Holding the target fixed removes the between-condition nominal target-distance difference, but neither holds realized carry fixed nor identifies a pure causal lean, curvature, or distance effect.')
add(r'\subsection{Conditional Synthesis and Limits of Inference}')
rows=[]
for club in clubnames:
 subset=[c for c in cells if c['config']['club']['club_id']==club]
 for pat in ['Draw','Fade']:
  vs=[c['course_scoring']['paired_benefit_vs_straight'][pat] for c in subset];pos=sum(v['lower_95']>0 for v in vs);neg=sum(v['upper_95']<0 for v in vs)
  rows.append([clubnames[club],pat,num(min(v['estimate'] for v in vs)),num(max(v['estimate'] for v in vs)),f'{pos}/{neg}/{8-pos-neg}'])
add(r'{\small');table('Selected Relative-SG Ranges and Pointwise Interval Directions',['Club','Pattern','Minimum','Maximum','Pos/Neg/Spans 0'],rows);add('}')
add(r'For the eight selected 7-Iron conditions, Fade intervals are positive and Draw intervals negative. Driver and PW have different signs and zero-spanning cases as tabulated. Consequently these selected modeled comparisons do not support a general recommendation to prefer one shape across clubs and delivery distributions. Counts of pointwise interval directions are descriptive, not multiplicity-adjusted discoveries. No interval here includes uncertainty in the physical model, course model, clubhead assumptions or human delivery distribution.')
add(r'The geometric derivative above establishes that face closure and delivered loft are coupled for this shaft-axis convention; it does not by itself establish carry direction. The earlier source-bound Straight examples show opposite lateral--downrange correlation signs for Driver versus the two irons. Their observed long-left/short-right or short-left/long-right patterns are local results of the complete impact and flight model. Marginal dispersion or carry changes cannot quantitatively decompose the SG difference into a distance effect and a shape effect.')
add(r'\subsection{Numerical Traceability and Publication Status}')
add(r'The selected 24 conditions and 72 rows are bound by \href{phase2/validation/selected_headline_traceability_v1/review.json}{Headline Traceability} to 52 summary/manifest hashes. The direct interactions and controls use the actual selectors in \texttt{final\_report\_inputs.py} and the \href{phase2/supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json}{384-Comparison Numerical Ledger}: 164 SD, 164 curve-strength and 56 across-lean comparisons. All seven numerical artifacts were byte-reconciled against \href{phase2/validation/protocol_name_recovery_v6/expected_numeric_hashes.json}{Expected Numerical Hashes}. This is numerical qualification only; it is not acceptance of the still-running graphical publication.')
add(r'The full matrix inventories remain 324 primary cells (270 completed, 54 unavailable; 8.1 million attempted rows), 72 original fixed-target cells (52 completed, 20 unavailable; 1.56 million rows), and 26 supplemental cells (780,000 rows, 779,712 supported and 288 unsupported). Four verified aliases add no shots. Source audits are \href{phase2/validation/independent_primary_final_audit.json}{Primary}, \href{phase2/validation/independent_control_final_audit.json}{Control}, and \href{phase2/validation/supplement_final_completion.json}{Supplemental Completion}. Unavailable bounded-search records are not relabeled as physical impossibility.')
add(r'All results remain conditional on assumed head geometry/inertia, the incremental Coulomb/energetic-restitution contact law, the numerical flight model, primitive Gaussian errors and scoring approximations described in the design reference. Gaussian tails are not certified by finite deterministic refinement grids. Shared seeded draws correlate cells; the pointwise bootstrap intervals express Monte Carlo sampling uncertainty, not independent empirical replications. The documented head-parameter and contact-law sensitivity is not a measured uncertainty distribution. Empirical qualification remains deferred under issue \#11984. Final graphics, full publication review and successful compilation of this editable article remain separate open gates.')
block='\n\n'.join(lines)+'\n\n'
s=s[:start]+block+s[end:]
s=s.replace(r'\usepackage{amsmath,booktabs,graphicx,hyperref}',r'\usepackage{amsmath,booktabs,graphicx,hyperref,longtable,textcomp}')
s=s.replace(r'\begin{document}',r'\newcommand{\numtext}[1]{\textnormal{#1}}'+'\n'+r'\begin{document}',1)
article.write_text(s)
(P/'generated_numeric_fields.json').write_text(json.dumps(checks,indent=2)+'\n')
(P/'repaired_article.tex').write_text(s)
print(hashlib.sha256(s.encode()).hexdigest(),len(checks))
