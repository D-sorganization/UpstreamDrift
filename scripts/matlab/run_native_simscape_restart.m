function receipt = run_native_simscape_restart(out_dir)
%RUN_NATIVE_SIMSCAPE_RESTART R2025b physical/discrete operating-point fixture.
%   Preconditions: licensed R2025b, writable OUT_DIR. This small generated
%   mass-spring-damper network is a restart prerequisite, not golfer evidence.
%   Postconditions: saved/reloaded complete ModelOperatingPoint and immutable
%   input rows reproduce native physical and discrete outputs after t = 0.2 s.

    arguments
        out_dir (1,1) string
    end
    assert(strcmp(version('-release'), '2025b'), 'R2025bRequired');
    if ~isfolder(out_dir); mkdir(out_dir); end
    mdl = 'native_restart_fixture_11921';
    if bdIsLoaded(mdl); close_system(mdl, 0); end
    cleanup = onCleanup(@() local_close_model(mdl)); %#ok<NASGU>
    local_build_model(mdl);
    model_path = fullfile(out_dir, mdl + ".slx");
    save_system(mdl, model_path);
    model_sha256 = native_simscape_file_sha256(model_path);

    input_time = (0:0.01:0.4).';
    input_force = 0.4 + 0.2 * (input_time >= 0.1) ...
        - 0.3 * (input_time >= 0.2) + 0.1 * (input_time >= 0.3);
    input_path = fullfile(out_dir, 'native-restart-input.mat');
    save(input_path, 'input_time', 'input_force', '-v7.3');
    saved_input_sha256 = local_hash_input(input_time, input_force);
    saved = load(input_path, 'input_time', 'input_force');
    assert(isequal(saved.input_time, input_time) && ...
        isequal(saved.input_force, input_force), 'SavedInputReadbackMismatch');
    replayed_input_sha256 = local_hash_input(saved.input_time, saved.input_force);

    base = Simulink.SimulationInput(mdl);
    base = setVariable(base, 'native_force_input', ...
        timeseries(input_force, input_time));
    full = sim(setModelParameter(base, 'StopTime', '0.4'));
    first = sim(setModelParameter(base, 'StopTime', '0.2'));
    op = first.xFinal;
    assert(isa(op, 'Simulink.op.ModelOperatingPoint'), ...
        'CompleteNativeOperatingPointRequired');
    actual_snapshot_time_s = double(op.snapshotTime);
    assert(abs(actual_snapshot_time_s - 0.2) < 1e-10 && ...
        abs(double(op.startTime)) < 1e-10, 'NativeSnapshotTimeMismatch');
    native_meta = struct('model_name', mdl, ...
        'model_sha256', model_sha256, ...
        'input_sha256', saved_input_sha256, ...
        'solver_id', get_param(mdl, 'Solver'), ...
        'runtime_release', version('-release'), ...
        'snapshot_time_s', actual_snapshot_time_s);
    snapshot_path = fullfile(out_dir, 'native-operating-point.mat');
    save(snapshot_path, 'op', 'native_meta', '-v7.3');
    clear op
    snapshot_sha256 = native_simscape_file_sha256(snapshot_path);
    expected = native_meta;
    expected.snapshot_sha256 = snapshot_sha256;
    expected.owned_directory = out_dir;
    restored = load_native_simscape_snapshot(snapshot_path, expected);

    replay_input = Simulink.SimulationInput(mdl);
    replay_input = setVariable(replay_input, 'native_force_input', ...
        timeseries(saved.input_force, saved.input_time));
    replay_input = setInitialState(replay_input, restored);
    replay = sim(setModelParameter(replay_input, 'StopTime', '0.4'));
    [physical_error, physical_samples] = local_compare_series( ...
        full.physical_position, replay.physical_position, 0.2);
    [discrete_error, discrete_samples] = local_compare_series( ...
        full.discrete_state, replay.discrete_state, 0.2);
    physical_span = max(double(full.physical_position.Data(:))) ...
        - min(double(full.physical_position.Data(:)));
    discrete_span = max(double(full.discrete_state.Data(:))) ...
        - min(double(full.discrete_state.Data(:)));
    assert(physical_span > 1e-6 && discrete_span > 0.1, ...
        'NativeFixtureMustBeDynamic');

    identity = struct('input_time', input_time, 'input_force', input_force, ...
        'input_path', input_path, 'model_sha256', model_sha256, ...
        'snapshot_sha256', snapshot_sha256, ...
        'saved_input_sha256', saved_input_sha256, ...
        'replayed_input_sha256', replayed_input_sha256, ...
        'snapshot_time_s', actual_snapshot_time_s);
    metrics = struct('physical_error', physical_error, ...
        'physical_samples', physical_samples, 'physical_span', physical_span, ...
        'discrete_error', discrete_error, ...
        'discrete_samples', discrete_samples, 'discrete_span', discrete_span);
    receipt = local_receipt(mdl, out_dir, full, replay, saved, identity, metrics);
    assert(physical_error <= receipt.position_bound_m && ...
        discrete_error <= receipt.discrete_bound, 'NativeReplayMismatch');
    fid = fopen(fullfile(out_dir, 'native-restart-receipt.json'), 'w');
    assert(fid > 0, 'ReceiptOpenFailed');
    fprintf(fid, '%s\n', jsonencode(receipt, 'PrettyPrint', true));
    fclose(fid);
end

function receipt = local_receipt(mdl, out_dir, full, replay, saved, identity, metrics)
    max_executed_input_error_n = max(local_executed_input_error( ...
        full.executed_force, identity.input_time, identity.input_force), ...
        local_executed_input_error(replay.executed_force, ...
        identity.input_time, identity.input_force));
    assert(max_executed_input_error_n <= 1e-12, ...
        'NativeExecutedInputMismatch');
    diagnostics = struct();
    diagnostics.full_physical = local_export_series(out_dir, ...
        'full-physical.csv', full.physical_position.Time, ...
        full.physical_position.Data);
    diagnostics.replay_physical = local_export_series(out_dir, ...
        'replay-physical.csv', replay.physical_position.Time, ...
        replay.physical_position.Data);
    diagnostics.full_discrete = local_export_series(out_dir, ...
        'full-discrete.csv', full.discrete_state.Time, full.discrete_state.Data);
    diagnostics.replay_discrete = local_export_series(out_dir, ...
        'replay-discrete.csv', replay.discrete_state.Time, ...
        replay.discrete_state.Data);
    diagnostics.executed_input = local_export_series(out_dir, ...
        'executed-input.csv', full.executed_force.Time, ...
        full.executed_force.Data);
    diagnostics.saved_input = local_export_series(out_dir, ...
        'saved-input.csv', saved.input_time, saved.input_force);
    receipt = struct();
    receipt.issue = '#11921';
    receipt.matlab_release = version('-release');
    receipt.matlab_version = version;
    receipt.fixture = mdl;
    receipt.snapshot_class = 'Simulink.op.ModelOperatingPoint';
    receipt.split_time_s = 0.2;
    receipt.actual_snapshot_time_s = identity.snapshot_time_s;
    receipt.actual_replay_start_s = double(replay.physical_position.Time(1));
    assert(abs(receipt.actual_replay_start_s - 0.2) < 1e-10, ...
        'NativeReplayStartMismatch');
    receipt.stop_time_s = 0.4;
    receipt.native_state_saved = true;
    receipt.native_state_reloaded = true;
    receipt.saved_input_sha256 = identity.saved_input_sha256;
    receipt.replayed_input_sha256 = identity.replayed_input_sha256;
    receipt.input_interpolation = 'zero_order_hold';
    receipt.input_after_final_value = get_param( ...
        [mdl '/Input'], 'OutputAfterFinalValue');
    receipt.max_executed_input_error_n = max_executed_input_error_n;
    receipt.diagnostic_files = diagnostics;
    receipt.solver_id = get_param(mdl, 'Solver');
    receipt.block_reduction = get_param(mdl, 'BlockReduction');
    receipt.input_converter_unit = get_param([mdl '/ToPhysical'], 'Unit');
    receipt.input_converter_filtering = get_param( ...
        [mdl '/ToPhysical'], 'FilteringAndDerivatives');
    receipt.max_physical_position_error_m = metrics.physical_error;
    receipt.physical_span_m = metrics.physical_span;
    receipt.position_bound_m = 1e-5;
    receipt.physical_samples = metrics.physical_samples;
    receipt.max_discrete_state_error = metrics.discrete_error;
    receipt.discrete_span = metrics.discrete_span;
    receipt.discrete_bound = 1e-12;
    receipt.discrete_samples = metrics.discrete_samples;
    receipt.model_sha256 = identity.model_sha256;
    receipt.snapshot_sha256 = identity.snapshot_sha256;
    receipt.input_file_sha256 = native_simscape_file_sha256(identity.input_path);
    receipt.scope = 'small native Simscape plus Unit Delay; no golfer qualification';
end

function error_max = local_executed_input_error(series, time_s, force_n)
    t = double(series.Time(:));
    observed = double(series.Data(:));
    expected = interp1(time_s, force_n, t, 'previous', 'extrap');
    assert(numel(t) >= 2 && all(isfinite(observed)) && ...
        all(t >= time_s(1)) && all(t <= time_s(end)), ...
        'NativeExecutedInputClockMismatch');
    error_max = max(abs(observed - expected));
end

function record = local_export_series(out_dir, filename, time_s, values)
    path = fullfile(out_dir, filename);
    t = double(time_s(:));
    v = double(values(:));
    assert(numel(t) == numel(v) && all(isfinite(t)) && ...
        all(isfinite(v)), 'NativeDiagnosticSeriesInvalid');
    writetable(table(t, v, 'VariableNames', {'time_s','value'}), path);
    record = struct('path', filename, ...
        'sha256', native_simscape_file_sha256(path));
end

function local_build_model(mdl)
    load_system('fl_lib'); load_system('nesl_utility'); load_system('simulink');
    new_system(mdl);
    names = {'Mass', 'Translational Spring', 'Translational Damper', ...
        'Ideal Force Source', 'Ideal Translational Motion Sensor', ...
        'Mechanical Translational Reference', 'Solver Configuration', ...
        'Simulink-PS Converter', 'PS-Simulink Converter'};
    tags = {'Mass','Spring','Damper','Force','Sensor','Reference', ...
        'Solver','ToPhysical','ToSimulink'};
    for k = 1:numel(names)
        lib = 'fl_lib';
        if k >= 7; lib = 'nesl_utility'; end
        matches = find_system(lib, 'LookUnderMasks', 'all', 'Name', names{k});
        assert(numel(matches) == 1, 'NativeBlockNotUnique: %s', names{k});
        add_block(matches{1}, [mdl '/' tags{k}], ...
            'Position', [100 + k*90, 100, 160 + k*90, 160]);
    end
    add_block('simulink/Sources/From Workspace', [mdl '/Input'], ...
        'VariableName', 'native_force_input', 'Interpolate', 'off', ...
        'OutputAfterFinalValue', 'Holding final value', ...
        'Position', [30, 20, 100, 50]);
    add_block('simulink/Discrete/Unit Delay', [mdl '/Delay'], ...
        'SampleTime', '0.01', 'InitialCondition', '-0.5', ...
        'Position', [150, 20, 210, 50]);
    add_block('simulink/Sinks/To Workspace', [mdl '/PositionLog'], ...
        'VariableName', 'physical_position', 'SaveFormat', 'Timeseries', ...
        'Position', [940, 30, 1010, 60]);
    add_block('simulink/Sinks/To Workspace', [mdl '/DelayLog'], ...
        'VariableName', 'discrete_state', 'SaveFormat', 'Timeseries', ...
        'Position', [270, 20, 340, 50]);
    add_block('simulink/Sinks/To Workspace', [mdl '/ExecutedInputLog'], ...
        'VariableName', 'executed_force', 'SaveFormat', 'Timeseries', ...
        'Position', [120, 60, 190, 90]);
    set_param([mdl '/Mass'], 'mass', '1');
    set_param([mdl '/Spring'], 'spr_rate', '40');
    set_param([mdl '/Damper'], 'D', '2');
    set_param([mdl '/Sensor'], 'reference', ...
        'foundation.enum.MeasurementReference.absolute', ...
        'position_port', 'true', 'velocity_port', 'false', ...
        'acceleration_port', 'false');
    set_param([mdl '/ToPhysical'], 'Unit', 'N', ...
        'FilteringAndDerivatives', 'zero');
    set_param([mdl '/ToSimulink'], 'Unit', 'm');
    local_wire_model(mdl);
    set_param(mdl, 'Solver', 'ode23t', 'StopTime', '0.4', 'MaxStep', '0.01', ...
        'RelTol', '1e-8', 'AbsTol', '1e-10', ...
        'BlockReduction', 'off', ...
        'SaveFinalState', 'on', 'SaveOperatingPoint', 'on', ...
        'FinalStateName', 'xFinal', 'ReturnWorkspaceOutputs', 'on');
end

function local_wire_model(mdl)
    p = struct();
    for tag = {'Mass','Spring','Damper','Force','Sensor','Reference', ...
            'Solver','ToPhysical','ToSimulink','Input','Delay', ...
            'PositionLog','DelayLog','ExecutedInputLog'}
        p.(tag{1}) = get_param([mdl '/' tag{1}], 'PortHandles');
    end
    add_line(mdl, p.Mass.LConn(1), p.Spring.RConn(1));
    add_line(mdl, p.Mass.LConn(1), p.Damper.RConn(1));
    add_line(mdl, p.Mass.LConn(1), p.Force.RConn(2));
    add_line(mdl, p.Mass.LConn(1), p.Sensor.LConn(1));
    add_line(mdl, p.Mass.LConn(1), p.Solver.RConn(1));
    add_line(mdl, p.Reference.LConn(1), p.Spring.LConn(1));
    add_line(mdl, p.Reference.LConn(1), p.Damper.LConn(1));
    add_line(mdl, p.Reference.LConn(1), p.Force.LConn(1));
    add_line(mdl, p.Input.Outport(1), p.ToPhysical.Inport(1));
    add_line(mdl, p.Input.Outport(1), p.ExecutedInputLog.Inport(1));
    add_line(mdl, p.Input.Outport(1), p.Delay.Inport(1));
    add_line(mdl, p.Delay.Outport(1), p.DelayLog.Inport(1));
    add_line(mdl, p.ToPhysical.RConn(1), p.Force.RConn(1));
    assert(numel(p.Sensor.RConn) == 1, 'PositionSensorPortMismatch');
    add_line(mdl, p.Sensor.RConn(1), p.ToSimulink.LConn(1));
    add_line(mdl, p.ToSimulink.Outport(1), p.PositionLog.Inport(1));
end

function local_close_model(mdl)
    if bdIsLoaded(mdl); close_system(mdl, 0); end
end

function [error_max, samples] = local_compare_series(reference, resumed, split_time)
    t = double(resumed.Time(:));
    assert(numel(t) >= 2 && all(t >= split_time - 1e-10), ...
        'NativeReplayClockMismatch');
    ref = interp1(double(reference.Time(:)), double(reference.Data(:)), ...
        t, 'linear');
    assert(all(isfinite(ref)) && all(isfinite(resumed.Data(:))), ...
        'NativeReplayNonfinite');
    error_max = max(abs(ref - double(resumed.Data(:))));
    samples = numel(t);
end

function hex = local_hash_input(time, force)
    values = [double(time(:)); double(force(:))];
    md = java.security.MessageDigest.getInstance('SHA-256');
    digest = typecast(md.digest(typecast(values, 'uint8')), 'uint8');
    hex = lower(reshape(dec2hex(digest, 2).', 1, []));
end
