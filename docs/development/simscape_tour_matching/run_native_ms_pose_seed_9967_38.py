"""Native MS pilot using shared solver; qualify derivatives before any fit."""
import argparse,hashlib,json
from src.shared.python.motion_matching.shooting_state_seed import select_shooting_states
from pathlib import Path
import numpy as np
from src.engines.physics_engines.pinocchio.python.native_model import NativePinocchioModel
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate,replay_window
from src.engines.physics_engines.pinocchio.python.native_sensitivity import replay_marker_sensitivities
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate,increment_native_bernstein,recover_native_bernstein
from src.shared.python.motion_matching.node_retraction import retract_node
from src.shared.python.motion_matching.native_restart import prepare_native_restart,widen_control_envelope
from src.shared.python.motion_matching.multi_shooting_fit import MultipleShootingOptions,fit_multiple_shooting
from src.shared.python.motion_matching.prefix_fit import MarkerTarget
from src.shared.python.motion_matching.shooting_schedule import sampled_shooting_windows
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--max-nfev',type=int,default=6);parser.add_argument('--defect-weight',type=float,default=100.);parser.add_argument('--audit-only',action='store_true');parser.add_argument('--horizon',type=float,default=.85);parser.add_argument('--nodes',type=float,nargs='+',default=[.2,.4,.6,.7,.8,.85]);parser.add_argument('--basis-duration',type=float,default=.8);parser.add_argument('--max-iterations',type=int,default=12);parser.add_argument('--equality-tolerance',type=float,default=1e-7);parser.add_argument('--node-bound',type=float,default=.02);parser.add_argument('--state-seed',type=Path,required=True);args=parser.parse_args()
assert np.isfinite(args.node_bound) and 0<args.node_bound and args.node_bound*np.sqrt(42)<=.5
assert args.nodes[-1]==args.horizon and np.isfinite(args.basis_duration) and args.basis_duration>0
assert np.isfinite(args.defect_weight) and args.defect_weight>0
root=Path('/mnt/c/Users/diete');out=Path(args.output);out.mkdir(exist_ok=False)
raw=(root/'native_geometry_spec_9967.json').read_bytes();spec=json.loads(raw);names=spec['coordinate_order'];n=len(names)
def load(p):return NativeReplayCandidate.from_document(json.loads(p.read_text()),names,hashlib.sha256(raw).hexdigest())
def covered(candidate):
 document=candidate.document;document['duration_s']=args.horizon
 return NativeReplayCandidate.from_document(document,names,hashlib.sha256(raw).hexdigest())
source_seed=load(root/'native-ms-fit-9967-19/returned-candidate.json')
assert source_seed.sha256=='b5b1c3823c86a21df323dc4e430366dc093495b069b3d25c361b0d007b8ff24f'
base=covered(load(root/'native-root-force-9967-02/returned-candidate.json'));seed=covered(source_seed)
payload=json.loads((root/'driver_marker_payload_9967.json').read_text());assert payload['source_sha256']==seed.document['capture_sha256']
windows=sampled_shooting_windows(np.array(payload['time_s']),args.nodes)
idx=[payload['labels'].index(x) for x in seed.document['marker_labels']];mask=np.array(payload['time_s'])<=args.horizon;t=np.array(payload['time_s'])[mask];points=np.array(payload['points_world_m'])[mask][:,idx];valid=np.array(payload['valid'],bool)[mask][:,idx];points[~valid]=np.nan
assert t[216]==.6 and t[-1]==args.horizon

prior_config=json.loads((root/'native-ms-fit-9967-04/config.json').read_text())
lower_effort=np.column_stack((np.full((n,2),-2.),np.array(prior_config['lower_effort']).reshape(n,5))).ravel();upper_effort=np.column_stack((np.full((n,2),2.),np.array(prior_config['upper_effort']).reshape(n,5))).ravel()
prior_run=root/'native-ms-fit-9967-19'
prior_run_config=json.loads((prior_run/'config.json').read_text())
bound_report=json.loads((prior_run/'bound-audit.json').read_text())
assert bound_report['candidate_sha256']==source_seed.sha256
assert np.array_equal(lower_effort,prior_run_config['lower_effort']) and np.array_equal(upper_effort,prior_run_config['upper_effort'])
selected=np.zeros((n,7),dtype=bool)
for item in bound_report['efforts']:
 selected[names.index(item['coordinate']),item['bernstein_index']]=True
recovered=recover_native_bernstein(base,seed,basis_duration_s=args.basis_duration).ravel()
actual=np.isclose(recovered,lower_effort,rtol=0,atol=1e-8)|np.isclose(recovered,upper_effort,rtol=0,atol=1e-8)
assert np.array_equal(selected.ravel(),actual) and selected.sum()==71
old_lower=lower_effort.copy();old_upper=upper_effort.copy()
lower_effort,upper_effort=[a.ravel() for a in widen_control_envelope(lower_effort.reshape(n,7),upper_effort.reshape(n,7),selected,factor=2.)]
policy={'qualification':'Explicit numerical control-bound continuation; not physical actuator limits','parent_candidate_sha256':source_seed.sha256,'bound_report_sha256':hashlib.sha256((prior_run/'bound-audit.json').read_bytes()).hexdigest(),'selected_count':int(selected.sum()),'selected_controls':np.argwhere(selected).tolist(),'factor':2.,'old_lower':old_lower.tolist(),'old_upper':old_upper.tolist(),'new_lower':lower_effort.tolist(),'new_upper':upper_effort.tolist(),'variable_scales_policy':'Frozen run19 vector; no scaling change with widened bounds','chart_centers_policy':'Run36 static pose and projected-rate interior guesses; original q0/qd0'}
(out/'bound-continuation.json').write_text(json.dumps(policy,indent=2)+'\n')
restart=prepare_native_restart(base,seed,basis_duration_s=args.basis_duration,lower_controls=lower_effort.reshape(n,7),upper_controls=upper_effort.reshape(n,7),roundoff_tolerance=1e-10)
seed=restart.candidate;initial=restart.controls.copy().ravel()
assert np.all(initial>=lower_effort) and np.all(initial<=upper_effort)
continuous=replay_candidate(raw,seed,t,rtol=1e-11,atol=1e-13,max_step=.00025)
source_replay=replay_candidate(raw,covered(source_seed),t,rtol=1e-11,atol=1e-13,max_step=.00025)
delta=continuous.integration.state-source_replay.integration.state
restart_audit={'qualification':'Restart reconstruction only; not fit acceptance','source_candidate_sha256':source_seed.sha256,'reconstructed_candidate_sha256':seed.sha256,'max_bound_snap':restart.max_bound_snap,'snapped_control_count':restart.snapped_control_count,'q_max_abs':float(np.max(abs(delta[:,:n]))),'qd_max_abs':float(np.max(abs(delta[:,n:]))),'marker_max_distance_m':float(np.max(np.linalg.norm(continuous.markers_m-source_replay.markers_m,axis=2)))}
restart_audit['passed']=restart_audit['q_max_abs']<=1e-6 and restart_audit['qd_max_abs']<=1e-4 and restart_audit['marker_max_distance_m']<=1e-7
(out/'restart-audit.json').write_text(json.dumps(restart_audit,indent=2)+'\n')
(out/'initial-candidate.json').write_text(json.dumps(seed.document,indent=2)+'\n')
# This separately identified candidate trial does not claim source-state parity.
effort_delta=max(float(np.max(abs(np.polyval(a,t)-np.polyval(b,t)))) for a,b in zip(seed.document['coefficients'],source_seed.document['coefficients']))
assert effort_delta<=1e-8 and restart_audit['q_max_abs']<=1e-6 and restart_audit['marker_max_distance_m']<=1e-7
(out/'distinct-seed-trial.json').write_text(json.dumps({'qualification':'Distinct reconstructed candidate trial; source-state comparison reported separately; no fit or engine parity gate changed','effort_profile_max_abs_difference':effort_delta,'source_state_parity_passed':restart_audit['passed'],'source_candidate_sha256':source_seed.sha256,'actual_seed_candidate_sha256':seed.sha256},indent=2)+'\n')
state_seed_raw=args.state_seed.read_bytes()
nodes=select_shooting_states(json.loads(state_seed_raw),args.nodes[:-1],model_sha256=hashlib.sha256(raw).hexdigest(),dimension=n)
engine=NativePinocchioModel(spec);d=np.r_[np.full(n,.1),np.ones(n)];cs=np.r_[np.full(6,.01),np.full(6,.1)]
def closure(x):
 engine.accelerations(dict(zip(names,x[:n])),dict(zip(names,x[n:])),dict.fromkeys(names,0.));return np.concatenate(engine.closure_errors())
def cj(x):
 h=1e-6;eye=np.eye(2*n)
 return np.column_stack([(closure(x+h*e)-closure(x-h*e))/(2*h) for e in eye])
bases={}
for time,node in nodes.items():
 assert np.max(abs(closure(node)))<1e-7
 _,sv,vt=np.linalg.svd(cj(node)*d/cs[:,None],full_matrices=True);assert sv[-1]>1e-6;bases[time]=vt[12:].T
node_cache={};state_lookup={};window_cache={}
def mapped(time,z):
 assert time in nodes
 key=(time,np.asarray(z).tobytes())
 if key not in node_cache:
  r=retract_node(nodes[time],bases[time],z,closure,cj,state_scales=d,residual_scales=cs,radius=.5,tolerance=1e-8);node_cache[key]=r;state_lookup[r.state.tobytes()]=r
 return node_cache[key]
def transform(time,z):return mapped(time,z).state
def transform_jac(time,z):return mapped(time,z).state_jacobian

def candidate(theta):
 controls=np.zeros((n,7));controls[:,:]=theta.reshape(n,7)
 assert np.all(theta>=lower_effort-1e-8) and np.all(theta<=upper_effort+1e-8)
 return increment_native_bernstein(base,controls,basis_duration_s=args.basis_duration)
def evaluate(theta,clock,state):
 c=candidate(theta);key=(c.sha256,clock.tobytes(),None if state is None else state.tobytes())
 if key not in window_cache:
  start=np.array(c.document['q0']+c.document['qd0']) if state is None else state
  tangent=None if state is None else state_lookup[state.tobytes()].state_jacobian
  r=replay_marker_sensitivities(raw,c,clock,first_control=0,basis_duration_s=args.basis_duration,initial_state=start,initial_sensitivity=tangent)
  j=r.marker_jacobian;end=r.state_jacobian[-1]
  if tangent is not None:
   inverse=np.linalg.pinv(tangent)
   j=np.concatenate((j[...,:189],j[...,189:]@inverse),axis=-1)
   end=np.concatenate((end[:,:189],end[:,189:]@inverse),axis=-1)
  if len(window_cache)>=8: window_cache.pop(next(iter(window_cache)))
  window_cache[key]=(r,j,end)
  with (out/'windows.jsonl').open('a') as f:f.write(json.dumps({'candidate_sha256':c.sha256,'start':float(clock[0]),'end':float(clock[-1]),'state_sha256':hashlib.sha256(start.tobytes()).hexdigest(),'sensitivity_s':r.sensitivity_elapsed_s,'primal_agreement_m':r.primal_marker_max_abs_difference_m})+'\n')
 return window_cache[key]
def segmented(theta,clock,state):
 r,_,_=evaluate(theta,clock,state);return r.replay.markers_m,r.replay.integration.state[-1]
def window_jac(theta,clock,state):
 _,j,e=evaluate(theta,clock,state);return j,e
def full(theta,clock):return replay_candidate(raw,candidate(theta),clock,rtol=1e-11,atol=1e-13,max_step=.00025).markers_m
def metrics(pred):
 e=np.sum((pred-points)**2,axis=2);early=valid&(t[:,None]<=.6);club=np.array([x.lower().startswith(('marker_2','marker_3')) for x in seed.document['marker_labels']]);wl,wr=[seed.document['marker_labels'].index(x) for x in ['WaistLeft','WaistRight']]
 vp=pred[-1,wr,:2]-pred[-1,wl,:2];vo=points[-1,wr,:2]-points[-1,wl,:2];yt=np.degrees(np.arctan2(vo[1],vo[0]));yp=np.degrees(np.arctan2(vp[1],vp[0]));yaw=abs((yp-yt+180)%360-180)/max(abs(yt),1)*100
 return dict(whole=float(np.sqrt(np.mean(e[valid]))),early=float(np.sqrt(np.mean(e[early]))),terminal=float(np.sqrt(np.mean(e[-1,valid[-1]]))),club=float(np.sqrt(np.mean(e[-1,club&valid[-1]]))),yaw_pct=float(yaw))
def accepted(pred):
 m=metrics(pred);return m['whole']<=.025 and m['early']<=.012 and m['terminal']<=.035 and m['club']<=.060 and m['yaw_pct']<=5
node_initial={time:np.zeros(42) for time in nodes}
variable_scales=np.array(prior_run_config['variable_scales']);assert variable_scales.shape==(189+sum(len(z) for z in node_initial.values()),)
config={'state_seed_sha256':hashlib.sha256(state_seed_raw).hexdigest(),'source_driver_sha256':'e3d448284682fb57e08a6ba8d54fc5265bdec25d8cd4a75ecaddf4a9c6990258','variable_scales':variable_scales.tolist(),'seed':seed.sha256,'base':base.sha256,'model_sha256':hashlib.sha256(raw).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'nodes':args.nodes,'horizon':args.horizon,'basis_duration_s':args.basis_duration,'source_seed_candidate_sha256':source_seed.sha256,'solver':'slsqp','max_iterations':args.max_iterations,'equality_tolerance':args.equality_tolerance,'callback_cost_kind':'marker residuals plus unweighted physical defects; solver objective excludes defects','first_control':0,'step_tolerance':None,'lower_effort':lower_effort.tolist(),'upper_effort':upper_effort.tolist(),'state_seed_candidate_sha256':seed.sha256,'state_seed_kind':'Run36 static pose and projected-rate interior guesses; original q0/qd0; final uninterrupted replay required','solver_source_sha256':hashlib.sha256(Path(fit_multiple_shooting.__code__.co_filename).read_bytes()).hexdigest(),'initial_effort_and_slope_fixed':False,'B0_B1_bound_policy':'See per-entry lower_effort/upper_effort and bound-continuation.json','node_box':[-args.node_bound,args.node_bound],'defect_scales':d.tolist(),'max_nfev':args.max_nfev,'defect_weight':args.defect_weight,'cache_capacity':8}
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
initial_defects={}
for start,end in [(float(w[0]),float(w[-1])) for w in windows[:-1]]:
 state=np.array(seed.document['q0']+seed.document['qd0']) if start==0 else nodes[start]
 r=replay_window(raw,seed,t[(t>=start)&(t<=end)],state,rtol=1e-11,atol=1e-13,max_step=.00025)
 initial_defects[str(end)]=float(np.linalg.norm((r.integration.state[-1]-nodes[end])/d))
(out/'initial-defects.json').write_text(json.dumps(initial_defects,indent=2)+'\n')
# Interior guesses may have initial defects; final acceptance gates are unchanged.
assert np.isfinite(list(initial_defects.values())).all()
# Qualification must complete before any optimization.
checks=[]
for time,end in [(float(w[0]),float(w[-1])) for w in windows[1:]]:
 clock=t[(t>=time)&(t<=end)];assert clock[0]==time and clock[-1]==end
 z=node_initial[time].copy();z[14]+=.8*args.node_bound;state=transform(time,z);mj,ej=window_jac(initial,clock,state);T=transform_jac(time,z)
 for col in [0,14]:
  delta=np.zeros(42);delta[col]=1e-5;pr=[];er=[]
  for sign in [-1,1]:
   x=transform(time,z+sign*delta);r=replay_window(raw,candidate(initial),clock,x,rtol=1e-11,atol=1e-13,max_step=.00025);pr.append(r.markers_m);er.append(r.integration.state[-1])
  fd=(pr[1]-pr[0])/2e-5;fe=(er[1]-er[0])/2e-5;ana=mj[...,189:]@T[:,col];ae=ej[:,189:]@T[:,col]
  checks.append({'time_s':time,'column':col,'marker_relative':float(np.linalg.norm(fd-ana)/np.linalg.norm(fd)),'state_relative':float(np.linalg.norm(fe-ae)/np.linalg.norm(fe))})
# Newly released B0/B1 columns must agree with native first-window differences.
clock=windows[0];mj,ej=window_jac(initial,clock,None)
early_columns=[next(col for col in range(control,189,7) if min(initial[col]-lower_effort[col],upper_effort[col]-initial[col])>1e-5) for control in [0,1]]
for col in early_columns:
 h=1e-5;delta=np.zeros(189);delta[col]=h;pr=[];er=[]
 for sign in [-1,1]:
  trial=candidate(initial+sign*delta);r=replay_window(raw,trial,clock,np.array(trial.document['q0']+trial.document['qd0']),rtol=1e-11,atol=1e-13,max_step=.00025);pr.append(r.markers_m);er.append(r.integration.state[-1])
 fd=(pr[1]-pr[0])/(2*h);fe=(er[1]-er[0])/(2*h)
 checks.append({'kind':'early_effort','coordinate':names[col//7],'bernstein_control':col%7,'column':col,'marker_relative':float(np.linalg.norm(fd-mj[...,col])/np.linalg.norm(fd)),'state_relative':float(np.linalg.norm(fe-ej[:,col])/np.linalg.norm(fe))})
passed=all(max(v['marker_relative'],v['state_relative'])<1e-3 for v in checks)
(out/'derivative-audit.json').write_text(json.dumps({'passed':passed,'checks':checks},indent=2)+'\n')
if not passed:raise ValueError('Composed derivative audit failed')
evaluation_count=0
config_hash=hashlib.sha256((out/'config.json').read_bytes()).hexdigest()
def checkpoint(theta,states,cost):
 global evaluation_count
 evaluation_count+=1
 c=candidate(theta);payload={'qualification':'residual evaluation only; no acceptance or convergence','evaluation':evaluation_count,'candidate_sha256':c.sha256,'candidate':c.document,'physical_nodes':{str(k):v.tolist() for k,v in states.items()},'residual_sum_squares':cost,'config_sha256':config_hash}
 path=out/f'evaluation-{evaluation_count:05d}.json';temp=path.with_suffix('.tmp')
 with temp.open('x') as stream:json.dump(payload,stream,indent=2);stream.write('\n')
 assert not path.exists();temp.replace(path)
 with (out/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps({'evaluation':evaluation_count,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cost':cost})+'\n')
if args.audit_only:raise SystemExit(0)

options=MultipleShootingOptions(solver='slsqp',variable_scales=variable_scales,max_iterations=args.max_iterations,equality_tolerance=args.equality_tolerance,constraint_projection=lambda time:bases[time].T,shooting_nodes=tuple(args.nodes),step_tolerance=None,max_nfev=args.max_nfev,defect_weight=args.defect_weight,defect_tolerance=1e-4,terminal_weight=10.,state_transform=transform,state_transform_jacobian=transform_jac,defect_scales=d,window_jacobian=window_jac,acceptance=accepted,checkpoint_callback=checkpoint)
fit=fit_multiple_shooting(MarkerTarget(t,points,np.ones(len(idx))),segmented,full,initial_theta=initial,lower_theta=lower_effort,upper_theta=upper_effort,initial_states=node_initial,state_bounds={time:(np.full(42,-args.node_bound),np.full(42,args.node_bound)) for time in nodes},options=options)
c=candidate(fit.theta);pred=full(fit.theta,t)
(out/'returned-candidate.json').write_text(json.dumps(c.document,indent=2)+'\n')
(out/'returned.json').write_text(json.dumps({'candidate_sha256':c.sha256,'metrics':metrics(pred),'accepted':fit.accepted,'optimizer_converged':fit.optimizer_converged,'max_scaled_defect_norm':fit.max_defect_norm,'message':fit.message,'defect_norms':fit.defect_norms,'optimality':fit.optimality,'function_evaluations':fit.function_evaluations,'active_bound_count':fit.active_bound_count,'terminal_replay_gap_m':fit.terminal_replay_gap_m,'previous_terminal_rms_m':float(np.sqrt(np.mean(np.sum((pred[t==.8]-points[t==.8])**2,axis=2)[valid[t==.8]]))),'segmented_rms':fit.segmented_rmse_m,'unsegmented_rms':fit.unsegmented_rmse_m},indent=2)+'\n')

(out/'returned-nodes.json').write_text(json.dumps({str(k):v.tolist() for k,v in fit.intermediate_states.items()},indent=2)+'\n')
