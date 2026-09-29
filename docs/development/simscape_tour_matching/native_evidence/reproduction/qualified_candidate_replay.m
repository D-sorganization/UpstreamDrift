function qualified_candidate_replay(repo,run_dir)
%QUALIFIED_CANDIDATE_REPLAY Independent cold replay of a saved torque-fit result.
assert(strcmp(version('-release'),'2025b'),'R2025b is required');
source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(source,'src/model'));addpath(genpath(fullfile(source,'src/functions')));addpath(fullfile(source,'motion_matching/shared'));
fit=jsondecode(fileread(fullfile(run_dir,'first_prefix_fit.json')));
assert(strcmp(fit.status,'exploratory-fit-computed') && fit.initial_state_verified);
saved=load(fullfile(run_dir,'final_native_replay.mat'),'fit_theta','fit_seed','fit_offsets');
seed=saved.fit_seed;assert(isequal(seed.q,fit.fit_identity.q) && isequal(seed.qd,fit.fit_identity.qd));
assert(isequal(seed.geometry_in,fit.fit_identity.geometry_in) && isequal(saved.fit_offsets,fit.fit_identity.offsets_m));
capture=jsondecode(fileread(fullfile(run_dir,'driver_marker_payload.json')));assert(strcmp(capture.source_sha256,fit.source_sha256));
clock=capture.time_s(capture.time_s<=fit.duration_s+1e-12);
load_system('GolfSwing3D_Kinetic');guard=configure_capture_velocity_targets();
ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');names={'UpperArmLength','LowerArmLength'};
for j=1:2;assignin(ws,names{j},seed.geometry_in(j));end
[ks,schema]=build_golf_kinematics();assert(isequal(string(seed.coordinate_names(:)),schema.coordinate_names(:)));
addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
opts=capture_fit_sim_options(fit.duration_s);opts.sample_rate=360;opts.fast_restart=false;opts.retain_raw_output=true;opts.verbosity='Silent';opts.joint_names=string(seed.coordinate_names)';
for j=1:2;opts.input_overrides.(names{j})=seed.geometry_in(j);end
for j=1:numel(seed.q)
 name=opts.joint_names(j);position=seed.q(j);velocity=seed.qd(j);
 if ~startsWith(name,'Translation');position=rad2deg(position);velocity=rad2deg(velocity);end
 opts.input_overrides.(replace(name,'Input','StartPosition'))=position;opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
[found,bodies]=ismember(string(seed.body_names),string({schema.frames.name}));assert(all(found));
[prediction,replay]=simulate_golf_markers(saved.fit_theta,opts,ks,schema,bodies(:),saved.fit_offsets,clock);
report=struct('matlab',version,'required_release','R2025b','qualification','independent cold prefix replay only', ...
 'duration_s',fit.duration_s,'source_sha256',fit.source_sha256,'marker_max_difference_m',max(abs(prediction-fit.final_prediction_m),[],'all'), ...
 'initial_q_max_error',max(abs(replay.q(1,:)-seed.q(:)')),'initial_qd_max_error',max(abs(replay.qd(1,:)-seed.qd(:)')), ...
 'fast_restart',get_param('GolfSwing3D_Kinetic','FastRestart'),'prediction_m',prediction);
fid=fopen(fullfile(run_dir,'qualified_candidate_replay.json'),'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
save(fullfile(run_dir,'qualified_candidate_replay.mat'),'report','prediction','replay','-v7.3');
assert(report.marker_max_difference_m<1e-8 && report.initial_q_max_error<1e-8 && report.initial_qd_max_error<1e-8,'Cold replay differs from saved candidate');
assert(strcmp(report.fast_restart,'off'));
clear guard;close_system('GolfSwing3D_Kinetic',0);bdclose('all');
end
