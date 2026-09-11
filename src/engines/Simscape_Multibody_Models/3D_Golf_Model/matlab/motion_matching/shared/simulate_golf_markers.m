function [markers,replay] = simulate_golf_markers(theta,opts,ks,schema,body_indices,offsets,time)
%SIMULATE_GOLF_MARKERS Native forward replay with fixed body marker attachments.
% Configure ks targets as schema.q_ids and outputs as frame_ids then rotation_ids.
% The caller must retain the same geometry as that kinematics snapshot. Every
% call starts through simulate_with_coefficients; no measured state is injected.
% Project at native sample times before Cartesian interpolation: never blend
% rotation matrices. Missing coverage or invalid solved states stop the fit.
    arguments
        theta (:,1) double {mustBeReal, mustBeFinite}
        opts (1,1) struct
        ks
        schema (1,1) struct
        body_indices (:,1) double {mustBeInteger, mustBePositive}
        offsets (:,3) double {mustBeReal, mustBeFinite}
        time (:,1) double {mustBeReal, mustBeFinite}
    end
    assert(numel(time)>=2 && time(1)==0 && all(diff(time)>0), ...
        'simulate_golf_markers:clock', 'Request increasing physical times from zero.');
    assert(time(end)<=opts.simulation_time, 'simulate_golf_markers:coverage', ...
        'Requested markers must be covered by the forward simulation.');
    opts.retain_raw_output = true;
    replay = simulate_with_coefficients(theta,opts);
    assert(strcmp(replay.solver_status,'success'), 'simulate_golf_markers:simulation', ...
        'Native simulation did not succeed.');
    bus = replay.raw_output.CombinedSignalBus;
    native_time = double(bus.HipLogs.HipPositionX.Time(:));
    native_joints = extract_golf_joint_kinematics(bus,schema.coordinate_names',native_time);
    assert(all(isfinite(native_joints.q),'all'), ...
        'simulate_golf_markers:coordinates', 'Native measured joint states are incomplete.');
    n = numel(schema.frames);
    m = numel(body_indices);
    native = zeros(numel(native_time),m,3);
    for k = 1:numel(native_time)
        [values,status,targets] = solve(ks,native_joints.q(k,:)');
        assert(status==1 && all(targets) && numel(values)==6*n && all(isfinite(values)), ...
            'simulate_golf_markers:kinematics', ...
            'Measured state did not resolve at t=%.17g: status=%d, targets=%d/%d, outputs=%d/%d.', ...
            native_time(k),status,nnz(targets),numel(targets),numel(values),6*n);
        positions = reshape(values(1:3*n),3,[])';
        rotations = intrinsic_xyz_to_rotm(reshape(values(3*n+1:end),3,[])');
        native(k,:,:) = reshape(project_body_markers(positions,rotations,body_indices,offsets),1,m,3);
    end
    flat = resample_logged_signal(reshape(native,numel(native_time),[]),time,3*m,native_time);
    assert(all(isfinite(flat),'all'), 'simulate_golf_markers:coverage', ...
        'Native marker output does not cover all requested physical times.');
    markers = reshape(flat,numel(time),m,3);
end
