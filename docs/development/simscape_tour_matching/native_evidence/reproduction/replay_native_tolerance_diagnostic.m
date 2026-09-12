function replay_native_tolerance_diagnostic(repo, source_path, output_path)
%REPLAY_NATIVE_TOLERANCE_DIAGNOSTIC Cold same-input solver qualification.
% Diagnostic only; use the shared forward wrapper and never save the model.
arguments
    repo (1,1) string
    source_path (1,1) string
    output_path (1,1) string
end
assert(strcmp(version('-release'),'2025b'),'Require R2025b');
assert(~isfile(output_path),'Refuse to overwrite evidence');
root=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
addpath(fullfile(root,'src/model'));
addpath(genpath(fullfile(root,'src/functions')));
addpath(fullfile(root,'motion_matching/shared'));
saved=load(source_path,'fit_theta','fit_seed','fit_offsets','fit_last_replay');
fit_theta=saved.fit_theta; fit_seed=saved.fit_seed; fit_offsets=saved.fit_offsets;
model='GolfSwing3D_Kinetic'; load_system(model);
cleanup=onCleanup(@() close_system(model,0));
guard=configure_capture_velocity_targets();
fields={'Solver','RelTol','AbsTol','MaxStep'};
original=struct();
for k=1:numel(fields); original.(fields{k})=get_param(model,fields{k}); end
set_param(model,'RelTol','1e-10','AbsTol','1e-12','MaxStep','0.0001');
bus=saved.fit_last_replay.raw_output.CombinedSignalBus;
clock=double(bus.HipLogs.HipPositionX.Time(:));
opts=capture_fit_sim_options(clock(end));
opts.sample_rate=360; opts.retain_raw_output=true; opts.verbosity='Silent';
opts.joint_names=string(fit_seed.coordinate_names)';
names={'UpperArmLength','LowerArmLength'};
for k=1:2; opts.input_overrides.(names{k})=fit_seed.geometry_in(k); end
for k=1:numel(fit_seed.q)
    name=opts.joint_names(k); q=fit_seed.q(k); v=fit_seed.qd(k);
    if ~startsWith(name,'Translation'); q=rad2deg(q); v=rad2deg(v); end
    opts.input_overrides.(replace(name,'Input','StartPosition'))=q;
    opts.input_overrides.(replace(name,'Input','StartVelocity'))=v;
end
started=tic;
fit_last_replay=simulate_with_coefficients(fit_theta,opts);
assert(strcmp(fit_last_replay.solver_status,'success'),'Replay failed');
report=struct('release',version('-release'),'source_path',source_path, ...
    'qualification','tight native diagnostic; parity not accepted', ...
    'elapsed_s',toc(started),'loaded_model_settings',original, ...
    'requested_settings',struct('Solver',opts.solver,'RelTol','1e-10', ...
    'AbsTol','1e-12','MaxStep','0.0001'));
save(output_path,'fit_theta','fit_seed','fit_offsets','fit_last_replay','report','-v7.3');
fid=fopen(output_path+'.json','w'); assert(fid~=-1);
closer=onCleanup(@() fclose(fid));
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));
end
