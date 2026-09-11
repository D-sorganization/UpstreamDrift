function cold_candidate_replay(repo, run_dir)
%COLD_CANDIDATE_REPLAY Qualify a saved candidate in an independent R2025b process.
assert(strcmp(version('-release'),'2025b'),'R2025b is required');
source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(source,'src/model')); addpath(genpath(fullfile(source,'src/functions')));
addpath(fullfile(source,'motion_matching/shared'));
input=jsondecode(fileread(fullfile(run_dir,'cold_candidate_input.json')));
opts=capture_fit_sim_options(input.time_s(end));
opts.sample_rate=360; opts.fast_restart=false; opts.retain_raw_output=true; opts.verbosity='Silent';
opts.joint_names=string(input.seed.coordinate_names)';
for j=1:numel(input.seed.q)
    name=opts.joint_names(j); value=input.seed.q(j);
    if ~startsWith(name,'Translation'); value=rad2deg(value); end
    opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
    opts.input_overrides.(replace(name,'Input','StartVelocity'))=0;
end
load_system('GolfSwing3D_Kinetic');
[ks,schema]=build_golf_kinematics();
addTargetVariables(ks,schema.q_ids); addOutputVariables(ks,schema.frame_ids); addOutputVariables(ks,schema.rotation_ids);
[found,bodies]=ismember(string(input.body_names),string({schema.frames.name})); assert(all(found));
started=tic;
[prediction,replay]=simulate_golf_markers(input.theta,opts,ks,schema,bodies(:),input.offsets,input.time_s);
report=struct('matlab',version,'required_release','R2025b','source_revision','9a381f018c2fe96b1b36c23a3c7d8c5aaeb3f74e', ...
    'qualification','independent replay of fixed 50 ms candidate only','elapsed_s',toc(started), ...
    'source_sha256',input.source_sha256,'marker_max_difference_m',max(abs(prediction-input.expected_prediction_m),[],'all'), ...
    'initial_q_max_error',max(abs(replay.q(1,:)-input.seed.q(:)')), ...
    'fast_restart',get_param('GolfSwing3D_Kinetic','FastRestart'),'prediction_m',prediction);
fid=fopen(fullfile(run_dir,'cold_candidate_replay.json'),'w'); assert(fid~=-1);
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
save(fullfile(run_dir,'cold_candidate_replay.mat'),'replay','prediction','report','-v7.3');
assert(report.marker_max_difference_m<1e-8,'Fresh replay differs from saved candidate');
assert(report.initial_q_max_error<1e-8,'Initial state changed');
assert(strcmp(report.fast_restart,'off'),'Expected cold simulation');
bdclose('all');
end

