function initial_velocity_seed(repo,run_dir,candidate_path,pose_path)
%INITIAL_VELOCITY_SEED Fit a native tangent velocity and verify forward initialization.
% Optional fixed-attachment candidate and initial pose preserve multiframe offsets.
% Native projection consistency is distinct from nonzero capture-marker error.
assert(strcmp(version('-release'),'2025b'));
base=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(base,'src/model'));addpath(genpath(fullfile(base,'src/functions')));addpath(fullfile(base,'motion_matching/shared'));
evidence=fullfile(repo,'docs/development/simscape_tour_matching/native_evidence');
fixed_attachments=nargin>=3;
if fixed_attachments
    assert(nargin==4,'Provide both attachment candidate and initial pose');
    candidate=jsondecode(fileread(candidate_path));seed=jsondecode(fileread(pose_path));
    assert(seed.frame_index==1 && seed.solver_flag==1 && all(seed.target_flags), ...
        'Initial pose must be a valid native capture frame 1');
    assert(isequal(seed.geometry_in(:),candidate.geometry_in(:)) && ...
        strcmp(seed.source_sha256,candidate.source_sha256),'Candidate identity differs');
    seed.parameter_values=candidate.geometry_in;fit=candidate;fixture=candidate;
    copyfile(candidate_path,fullfile(run_dir,'attachment_candidate.json'));
    copyfile(pose_path,fullfile(run_dir,'initial_pose.json'));
    qualification='initial state only; fixed multiframe attachments; capture residual reported separately';
else
    sweep=jsondecode(fileread(fullfile(evidence,'native_geometry_seed_sweep_r2025b.json')));rows=sweep.results;if iscell(rows);rows=vertcat(rows{:});end
    [~,index]=min([rows.proxy_euclidean_rms_m]);seed=rows(index);
    fit=jsondecode(fileread(fullfile(evidence,'first_prefix_fit_r2025b.json')));
    fixture=jsondecode(fileread(fullfile(evidence,'reproduction/cold_candidate_input.json')));
    qualification='initial velocity seed only; single-frame offsets; geometry provisional';
end
capture=jsondecode(fileread(fullfile(run_dir,'driver_marker_payload.json')));
if fixed_attachments;assert(strcmp(candidate.source_sha256,capture.source_sha256),'Capture identity differs');end
[found,columns]=ismember(string(fit.labels),string(capture.labels));assert(all(found));
window=1:7;assert(all(capture.valid(window,columns),'all'));
observed0=squeeze(capture.points_world_m(1,columns,:));
series=reshape(permute(capture.points_world_m(window,columns,:),[1 3 2]),numel(window),[]);
coefficients=[ones(numel(window),1),capture.time_s(window)]\series;desired=coefficients(2,:)';
load_system('GolfSwing3D_Kinetic');priority_cleanup=configure_capture_velocity_targets();ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
names={'UpperArmLength','LowerArmLength'};original=cellfun(@(name)getVariable(ws,name),names,'UniformOutput',false);cleanup=onCleanup(@()restore(ws,names,original));
for j=1:2;assignin(ws,names{j},seed.parameter_values(j));end
[ks,schema]=build_golf_kinematics();[found,order]=ismember(schema.coordinate_names,string(seed.coordinate_names));assert(all(found));q=seed.q(order);
addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
[values,status,targets]=solve(ks,q);assert(status==1 && all(targets));n=numel(schema.frames);
origins=reshape(values(1:3*n),3,[])';rotation=intrinsic_xyz_to_rotm(reshape(values(3*n+1:end),3,[])');
[found,bodies]=ismember(string(fixture.body_names),string({schema.frames.name}));assert(all(found));
if fixed_attachments
    offsets=candidate.offsets_m;
else
    offsets=zeros(size(observed0));
    for m=1:numel(bodies);b=bodies(m);offsets(m,:)=(observed0(m,:)-origins(b,:))*rotation(:,:,b);end
end
expected_initial=project_body_markers(origins,rotation,bodies(:),offsets);
map=golf_marker_velocity_map(ks,schema,q,bodies,offsets);
damping=1e-3;dimension=size(map.marker_jacobian,2);
rates=[map.marker_jacobian;damping*eye(dimension)]\[desired;zeros(dimension,1)];qd=map.joint_velocity_map*rates;
actual_velocity=map.marker_jacobian*rates;errors=reshape(actual_velocity-desired,3,[])';
report=struct('matlab',version,'source_revision','see archived run source','source_patches',{{'initial_velocity_seed.m'}}, ...
    'qualification',qualification,'source_sha256',capture.source_sha256, ...
    'geometry_in',seed.parameter_values,'labels',{fit.labels},'body_names',{fixture.body_names},'offsets_m',offsets, ...
    'coordinate_names',schema.coordinate_names,'q',q,'qd',qd,'independent_rates',rates, ...
    'window_end_s',capture.time_s(window(end)),'damping',damping,'map_rank',rank(map.marker_jacobian),'singular_values',svd(map.marker_jacobian), ...
    'target_velocity_m_s',reshape(desired,3,[])','fitted_velocity_m_s',reshape(actual_velocity,3,[])', ...
    'target_velocity_rms_m_s',sqrt(mean(sum(reshape(desired,3,[])'.^2,2))),'velocity_residual_rms_m_s',sqrt(mean(sum(errors.^2,2))));
write_report(run_dir,report);
clearTargetVariables(ks);clearOutputVariables(ks);clearInitialGuessVariables(ks);addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
opts=capture_fit_sim_options(0.05);opts.sample_rate=360;opts.fast_restart=false;opts.retain_raw_output=true;opts.verbosity='Silent';opts.joint_names=schema.coordinate_names';
for j=1:2;opts.input_overrides.(names{j})=seed.parameter_values(j);end
for j=1:numel(q)
    name=schema.coordinate_names(j);position=q(j);velocity=qd(j);
    if ~startsWith(name,'Translation');position=rad2deg(position);velocity=rad2deg(velocity);end
    opts.input_overrides.(replace(name,'Input','StartPosition'))=position;opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
[prediction,replay]=simulate_golf_markers(zeros(189,1),opts,ks,schema,bodies,offsets,[0;0.025;0.05]);
report.initial_q_max_error=max(abs(replay.q(1,:)-q'));
report.initial_qd_max_error=max(abs(replay.qd(1,:)-qd'));
report.initial_markers_max_error_m=max(abs(squeeze(prediction(1,:,:))-observed0),[],'all');
report.initial_projection_max_error_m=max(abs(squeeze(prediction(1,:,:))-expected_initial),[],'all');
target_errors=sqrt(sum((squeeze(prediction(1,:,:))-observed0).^2,2));
report.initial_target_rms_m=sqrt(mean(target_errors.^2));report.initial_target_max_error_m=max(target_errors);
report.initial_kinematic_markers_m=expected_initial;
report.prediction_m=prediction;report.status='replayed';write_report(run_dir,report);
save(fullfile(run_dir,'initial_velocity_seed.mat'),'map','replay','prediction','report','-v7.3');
assert(report.initial_q_max_error<1e-8 && report.initial_qd_max_error<1e-8 && report.initial_projection_max_error_m<1e-8,'Native initial state differs from the seed');
report.status='initial-state-qualified';report.initial_state_verified=true;write_report(run_dir,report);
save(fullfile(run_dir,'initial_velocity_seed.mat'),'report','-append');
clear priority_cleanup;clear cleanup;close_system('GolfSwing3D_Kinetic',0);
end
function write_report(run_dir,report)
fid=fopen(fullfile(run_dir,'initial_velocity_seed.json'),'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));fclose(fid);
end
function restore(ws,names,values)
for j=1:numel(names);assignin(ws,names{j},values{j});end
end
