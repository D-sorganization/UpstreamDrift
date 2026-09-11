function stitched_full_swing_replay(repo,stitched_dir,seed_path,payload_path)
%STITCHED_FULL_SWING_REPLAY Cold forward replay of continuous stitched full swing from t=0.
% Enforces single continuous simulation from initial state without resets.
assert(strcmp(version('-release'),'2025b'),'R2025b is required');
source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(source,'src/model'));addpath(genpath(fullfile(source,'src/functions')));addpath(fullfile(source,'motion_matching/shared'));

stitched=jsondecode(fileread(fullfile(stitched_dir,'stitched_swing.json')));
saved=load(fullfile(stitched_dir,'stitched_swing.mat'),'global_theta');
seed=jsondecode(fileread(seed_path));
capture=jsondecode(fileread(payload_path));

t_final=stitched.t_final_s;
clock=capture.time_s(capture.time_s<=t_final+1e-12);

load_system('GolfSwing3D_Kinetic');guard=configure_capture_velocity_targets();
ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');names={'UpperArmLength','LowerArmLength'};
for j=1:2;assignin(ws,names{j},seed.geometry_in(j));end
[ks,schema]=build_golf_kinematics();
addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);

opts=capture_fit_sim_options(t_final+0.010);
opts.sample_rate=360;opts.fast_restart=false;opts.retain_raw_output=true;opts.verbosity='Silent';
opts.joint_names=string(seed.coordinate_names)';
for j=1:2;opts.input_overrides.(names{j})=seed.geometry_in(j);end
for j=1:numel(seed.q)
 name=opts.joint_names(j);position=seed.q(j);velocity=seed.qd(j);
 if ~startsWith(name,'Translation');position=rad2deg(position);velocity=rad2deg(velocity);end
 opts.input_overrides.(replace(name,'Input','StartPosition'))=position;
 opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end

[found,bodies]=ismember(string(seed.body_names),string({schema.frames.name}));assert(all(found));
offsets=seed.offsets_m;
[prediction,replay]=simulate_golf_markers(saved.global_theta(:),opts,ks,schema,bodies(:),offsets,clock);

report=struct('matlab',version,'required_release','R2025b','qualification','continuous single-initial-state full swing replay', ...
 't_top_s',stitched.t_top_s,'t_final_s',t_final, ...
 'c0_continuous',stitched.continuity.c0_continuous,'max_c0_jump',stitched.continuity.max_c0_jump, ...
 'fast_restart',get_param('GolfSwing3D_Kinetic','FastRestart'),'prediction_m',prediction);

fid=fopen(fullfile(stitched_dir,'stitched_replay_report.json'),'w');assert(fid~=-1);
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
save(fullfile(stitched_dir,'stitched_replay.mat'),'report','prediction','replay','-v7.3');
clear guard;close_system('GolfSwing3D_Kinetic',0);bdclose('all');
fprintf('Continuous full-swing replay completed successfully through t=%.3f s!\n', t_final);
end
