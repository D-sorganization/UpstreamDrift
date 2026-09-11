function geometry_candidate_replay(repo,run_dir)
%GEOMETRY_CANDIDATE_REPLAY Verify changed geometry against native logged frames.
assert(strcmp(version('-release'),'2025b'));
base=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(base,'src/model'));addpath(genpath(fullfile(base,'src/functions')));addpath(fullfile(base,'motion_matching/shared'));
sweep=jsondecode(fileread(fullfile(repo,'docs/development/simscape_tour_matching/native_evidence/native_geometry_seed_sweep_r2025b.json')));
rows=sweep.results;if iscell(rows);rows=vertcat(rows{:});end
[~,index]=min([rows.proxy_euclidean_rms_m]);seed=rows(index);
load_system('GolfSwing3D_Kinetic');ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
names={'UpperArmLength','LowerArmLength'};original=cellfun(@(name)getVariable(ws,name),names,'UniformOutput',false);
cleanup=onCleanup(@()restore(ws,names,original));
opts=capture_fit_sim_options(0.05);opts.sample_rate=360;opts.fast_restart=false;opts.retain_raw_output=true;opts.verbosity='Silent';
for j=1:2;assignin(ws,names{j},seed.parameter_values(j));opts.input_overrides.(names{j})=seed.parameter_values(j);end
[ks,schema]=build_golf_kinematics();addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
opts.joint_names=string(seed.coordinate_names)';
for j=1:numel(seed.q)
    name=opts.joint_names(j);value=seed.q(j);if ~startsWith(name,'Translation');value=rad2deg(value);end
    opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
    opts.input_overrides.(replace(name,'Input','StartVelocity'))=0;
end
clock=[0;0.025;0.05];n=numel(schema.frames);started=tic;
[prediction,replay]=simulate_golf_markers(zeros(189,1),opts,ks,schema,(1:n)',zeros(n,3),clock);
measured=nan(3,n,3);errors=zeros(n,1);
for f=1:n
    if strlength(string(schema.frames(f).source))==0;continue;end
    signal=replay.raw_output.CombinedSignalBus;
    for key=split(string(schema.frames(f).source),'.')';signal=signal.(key);end
    expected=resample_logged_signal(signal,clock,3,[]);
    measured(:,f,:)=reshape(expected,3,1,3);
    errors(f)=max(abs(squeeze(prediction(:,f,:))-expected),[],'all');
end
pairs=["LS","LE";"LE","LF";"LF","LW";"RS","RE";"RE","RF";"RF","RW"];
lengths=zeros(6,1);
for k=1:6
    [found,ids]=ismember(pairs(k,:),string({schema.frames.name}));assert(all(found));
    lengths(k)=norm(squeeze(measured(1,ids(1),:))-squeeze(measured(1,ids(2),:)));
end
report=struct('matlab',version,'source_revision','317cc5d8d','qualification','changed-geometry zero-effort 50 ms replay only', ...
    'parameter_values_in',seed.parameter_values,'elapsed_s',toc(started),'initial_q_max_error',max(abs(replay.q(1,:)-seed.q(:)')), ...
    'frame_max_errors_m',errors,'frame_max_error_m',max(errors),'measured_arm_lengths_m',lengths,'frame_names',string({schema.frames.name}), ...
    'initial_positions_max_error_m',max(abs(squeeze(prediction(1,:,:))-seed.frame_positions_m),[],'all'));
fid=fopen(fullfile(run_dir,'geometry_candidate_replay.json'),'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
save(fullfile(run_dir,'geometry_candidate_replay.mat'),'replay','prediction','report','-v7.3');
assert(report.initial_q_max_error<1e-8 && report.initial_positions_max_error_m<1e-8,'Initial pose was not reproduced');
assert(report.frame_max_error_m<1e-8,'Native logs and geometry snapshot disagree');
expected=.0254*[seed.parameter_values(1);seed.parameter_values(2)/2;seed.parameter_values(2)/2];
assert(max(abs(lengths-[expected;expected]))<1e-8,'Forward geometry does not match requested lengths');
clear cleanup;close_system('GolfSwing3D_Kinetic',0);
end
function restore(ws,names,values)
for j=1:numel(names);assignin(ws,names{j},values{j});end
end
