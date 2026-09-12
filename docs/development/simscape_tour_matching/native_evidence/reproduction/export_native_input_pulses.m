function export_native_input_pulses(runtime_root,seed_path,spec_path,output_dir)
%EXPORT_NATIVE_INPUT_PULSES Baseline and 27 stationary input-response cases.
% Each case starts from the same requested pose and zero requested rates.
% Actual assembled states, native actuator audit and qdd are preserved.
    arguments
        runtime_root (1,1) string
        seed_path (1,1) string
        spec_path (1,1) string
        output_dir (1,1) string
    end
    assert(strcmp(version('-release'),'2025b'),'Require MATLAB R2025b.');
    assert(~isfolder(output_dir),'Refuse to overwrite an existing pulse batch.');
    mkdir(output_dir);
    root = fullfile(runtime_root,'src','engines','Simscape_Multibody_Models', ...
        '3D_Golf_Model','matlab');
    addpath(fullfile(root,'src','model'));
    addpath(genpath(fullfile(root,'src','functions')));
    addpath(fullfile(root,'motion_matching','shared'));
    seed = jsondecode(fileread(seed_path));
    spec = jsondecode(fileread(spec_path));
    names = string(seed.coordinate_names(:))';
    assert(isequal(names,string(spec.coordinate_order(:))'),'Coordinate identity mismatch.');
    load_system('GolfSwing3D_Kinetic');
    model_cleanup = onCleanup(@() close_system('GolfSwing3D_Kinetic',0));
    priority_cleanup = configure_capture_velocity_targets(); %#ok<NASGU>
    opts = capture_fit_sim_options(0.002);
    opts.sample_rate = 1000;
    opts.fast_restart = true;
    opts.retain_raw_output = true;
    opts.verbosity = 'Silent';
    opts.joint_names = names;
    geometry_names = ["UpperArmLength","LowerArmLength"];
    for j=1:2
        opts.input_overrides.(geometry_names(j)) = seed.geometry_in(j);
    end
    for j=1:numel(names)
        position = seed.q(j);
        if ~startsWith(names(j),'Translation')
            position = rad2deg(position);
        end
        opts.input_overrides.(replace(names(j),'Input','StartPosition')) = position;
        opts.input_overrides.(replace(names(j),'Input','StartVelocity')) = 0;
    end
    world_to_base = spec.joints(1).parent_to_base(1:3,1:3)';
    for case_index=0:numel(names)
        coefficients = zeros(numel(names),7);
        command = zeros(numel(names),1);
        if case_index>0
            command(case_index) = 1; % One N or Nm, by named input channel.
        end
        coefficients(:,7) = command;
        theta = reshape(coefficients',[],1);
        replay = simulate_with_coefficients(theta,opts);
        assert(strcmp(replay.solver_status,'success'),'Native pulse simulation failed.');
        bus = replay.raw_output.CombinedSignalBus;
        clock = double(bus.HipLogs.HipPositionX.Time(:));
        states = extract_golf_joint_kinematics(bus,names,clock);
        audit = audit_golf_actuator_torques(bus,theta,names,world_to_base);
        assert(isempty(audit.unlogged_coordinates),'Missing native actuator coverage.');
        assert(all(isfinite([states.q(1,:),states.qd(1,:),states.qdd(1,:)])), ...
            'Missing native initial dynamics samples.');
        primitive_efforts = command;
        primitive_efforts(1:3) = world_to_base*command(1:3);
        result = struct('release',version('-release'),'case_index',case_index, ...
            'coordinate_names',names,'requested_q',seed.q,'requested_qd',zeros(size(seed.q)), ...
            'q',states.q(1,:),'qd',states.qd(1,:),'qdd',states.qdd(1,:), ...
            'time_s',clock(1),'world_input',command,'primitive_efforts',primitive_efforts, ...
            'actuator_audit',audit,'geometry_in',seed.geometry_in);
        path = fullfile(output_dir,sprintf('case-%02d.json',case_index));
        fid = fopen(path,'w');
        assert(fid>=0,'Cannot open pulse output.');
        file_cleanup = onCleanup(@() fclose(fid));
        fprintf(fid,'%s\n',jsonencode(result));
        clear file_cleanup;
        fprintf('Completed native input case %d/%d\n',case_index,numel(names));
    end
    set_param('GolfSwing3D_Kinetic','FastRestart','off');
end
