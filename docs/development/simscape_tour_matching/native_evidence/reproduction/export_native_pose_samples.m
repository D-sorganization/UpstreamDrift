function export_native_pose_samples(runtime_root,replay_path,output_path)
%EXPORT_NATIVE_POSE_SAMPLES Independent native FK at exact saved rollout states.
% This diagnostic never saves or changes the source model on disk.
    arguments
        runtime_root (1,1) string
        replay_path (1,1) string
        output_path (1,1) string
    end
    assert(strcmp(version('-release'),'2025b'),'Require MATLAB R2025b.');
    assert(~isfile(output_path),'Refuse to overwrite an existing receipt.');
    root = fullfile(runtime_root,'src','engines','Simscape_Multibody_Models', ...
        '3D_Golf_Model','matlab');
    addpath(fullfile(root,'src','model'));
    addpath(genpath(fullfile(root,'src','functions')));
    addpath(fullfile(root,'motion_matching','shared'));
    saved = load(replay_path,'fit_last_replay','fit_seed');
    seed = saved.fit_seed;
    load_system('GolfSwing3D_Kinetic');
    cleanup = onCleanup(@() close_system('GolfSwing3D_Kinetic',0));
    workspace = get_param('GolfSwing3D_Kinetic','ModelWorkspace');
    assignin(workspace,'UpperArmLength',seed.geometry_in(1));
    assignin(workspace,'LowerArmLength',seed.geometry_in(2));
    [ks,schema] = build_golf_kinematics();
    addTargetVariables(ks,schema.q_ids);
    addOutputVariables(ks,schema.frame_ids);
    addOutputVariables(ks,schema.rotation_ids);
    bus = saved.fit_last_replay.raw_output.CombinedSignalBus;
    clock = double(bus.HipLogs.HipPositionX.Time(:));
    joints = extract_golf_joint_kinematics(bus,schema.coordinate_names',clock);
    requested = [0,0.4,0.6,0.7,0.75,0.8];
    assert(clock(end)>=requested(end)-1e-9,'Replay does not cover 0.8 seconds.');
    indices = zeros(size(requested));
    for k=1:numel(requested)
        [~,indices(k)] = min(abs(clock-requested(k)));
    end
    indices = unique(indices,'stable');
    n = numel(schema.frames);
    poses = zeros(numel(indices),n,4,4);
    for k=1:numel(indices)
        [values,status,targets] = solve(ks,joints.q(indices(k),:)');
        assert(status==1 && all(targets) && numel(values)==6*n ...
            && all(isfinite(values)),'Native pose solve failed.');
        positions = reshape(values(1:3*n),3,[])';
        rotations = intrinsic_xyz_to_rotm(reshape(values(3*n+1:end),3,[])');
        for f=1:n
            transform = eye(4);
            transform(1:3,1:3) = rotations(:,:,f);
            transform(1:3,4) = positions(f,:)';
            poses(k,f,:,:) = transform;
        end
    end
    result = struct('qualification','same-state FK only; dynamics unqualified', ...
        'release',version('-release'),'replay_path',replay_path, ...
        'time_s',clock(indices),'coordinate_names',schema.coordinate_names, ...
        'q',joints.q(indices,:),'frame_names',string({schema.frames.name}), ...
        'poses',poses,'geometry_in',seed.geometry_in);
    fid = fopen(output_path,'w');
    assert(fid>=0,'Cannot open output.');
    file_cleanup = onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(result));
end
