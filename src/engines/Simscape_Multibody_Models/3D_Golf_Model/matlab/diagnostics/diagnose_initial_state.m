function report = diagnose_initial_state(input_file, opts)
%DIAGNOSE_INITIAL_STATE  Compare specified vs constraint-resolved initial pose.
%
%   report = diagnose_initial_state(INPUT_FILE) reads INPUT_FILE
%   (e.g. "3DModelInputs_Impact.mat" or any input MAT), extracts the
%   specified initial joint angles, runs the Simscape model for one
%   solver step (StopTime=0, IC solve only), and returns a comparison
%   report quantifying how the loop-closure constraints (both arms
%   gripping the club, plus the club shaft loop) project the requested
%   pose onto the constraint manifold.
%
%   report = diagnose_initial_state(INPUT_FILE, OPTS) accepts an options
%   struct with fields:
%       .model_name     - Simulink model (default: "GolfSwing3D_KineticallyDriven")
%       .save_to        - optional MAT path to write the report to
%       .joint_threshold_deg - significance threshold (default 1 deg)
%       .pos_threshold_mm    - significance threshold (default 5 mm)
%
%   report fields:
%     .specified.q           - joint angles documented in the input file
%     .specified.r_butt      - documented Cartesian butt position (if available)
%     .specified.r_clubhead  - documented Cartesian clubhead position
%     .actual.q              - joint angles AFTER constraint projection
%     .actual.r_butt         - butt position after the model settles
%     .actual.r_clubhead     - clubhead position after the model settles
%     .delta.q_per_joint_deg - per-joint delta in degrees
%     .delta.q_max_deg       - largest single-joint delta
%     .delta.r_butt_mm       - Cartesian butt delta in mm (norm)
%     .delta.r_clubhead_mm   - Cartesian clubhead delta in mm (norm)
%     .delta.is_significant  - true if any joint > threshold or any pos > threshold
%     .input_file_hash       - sha256 for provenance
%     .input_file            - absolute path to the input file
%     .joint_names           - cell array of joint names in q
%     .timestamp             - ISO8601 timestamp of this diagnosis

    arguments
        input_file (1,1) string
        opts.model_name (1,1) string = "GolfSwing3D_KineticallyDriven"
        opts.save_to (1,1) string = ""
        opts.joint_threshold_deg (1,1) double = 1.0
        opts.pos_threshold_mm (1,1) double = 5.0
    end

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, 'private'));

    if ~isfile(input_file)
        error('diagnose_initial_state:InputNotFound', ...
            'Input file not found: %s', input_file);
    end

    % --- 1. Decode the SPECIFIED pose from the input MAT ---------------
    specified = decode_input_file_pose(input_file);

    % --- 2. Run the model for an IC-only solve -------------------------
    sim_in = Simulink.SimulationInput(opts.model_name);
    sim_in = sim_in.setModelParameter('StopTime', '0');
    sim_in = sim_in.setModelParameter('SaveFinalState', 'on');
    sim_in = sim_in.setModelParameter('SaveOperatingPoint', 'on');
    sim_in = sim_in.setModelParameter('SaveCompleteFinalSimState', 'on');
    sim_in = sim_in.setModelParameter('LoadInitialState', 'off');
    sim_in = sim_in.setVariable('GolfInputs', specified.raw_inputs);

    sim_out = sim(sim_in);

    % --- 3. Extract the ACTUAL converged pose --------------------------
    actual = extract_actual_pose(sim_out, specified.joint_names);

    % --- 4. Compute deltas ---------------------------------------------
    q_delta_rad = wrapToPi(actual.q - specified.q);
    q_delta_deg = rad2deg(q_delta_rad);

    delta = struct();
    delta.q_per_joint_deg = q_delta_deg;
    delta.q_max_deg = max(abs(q_delta_deg));

    if isfield(specified, 'r_butt') && ~isempty(specified.r_butt) ...
            && isfield(actual, 'r_butt') && ~isempty(actual.r_butt)
        delta.r_butt_mm = 1000 * norm(actual.r_butt - specified.r_butt);
    else
        delta.r_butt_mm = NaN;
    end

    if isfield(specified, 'r_clubhead') && ~isempty(specified.r_clubhead) ...
            && isfield(actual, 'r_clubhead') && ~isempty(actual.r_clubhead)
        delta.r_clubhead_mm = 1000 * norm(actual.r_clubhead - specified.r_clubhead);
    else
        delta.r_clubhead_mm = NaN;
    end

    significant = delta.q_max_deg > opts.joint_threshold_deg ...
        || (~isnan(delta.r_butt_mm) && delta.r_butt_mm > opts.pos_threshold_mm) ...
        || (~isnan(delta.r_clubhead_mm) && delta.r_clubhead_mm > opts.pos_threshold_mm);
    delta.is_significant = significant;

    % --- 5. Provenance hash --------------------------------------------
    file_hash = local_sha256(input_file);

    % --- 6. Assemble report --------------------------------------------
    report = struct();
    report.specified = specified;
    report.actual = actual;
    report.delta = delta;
    report.input_file = char(input_file);
    report.input_file_hash = file_hash;
    report.joint_names = specified.joint_names;
    report.model_name = char(opts.model_name);
    report.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX', 'TimeZone', 'local'));
    report.thresholds = struct( ...
        'joint_threshold_deg', opts.joint_threshold_deg, ...
        'pos_threshold_mm', opts.pos_threshold_mm);

    if opts.save_to ~= ""
        save(opts.save_to, '-struct', 'report');
    end
end

function h = local_sha256(filepath)
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(filepath, 'rb');
    cleanup = onCleanup(@() fclose(fid));
    while ~feof(fid)
        chunk = fread(fid, 65536, '*uint8');
        if ~isempty(chunk)
            md.update(chunk);
        end
    end
    bytes = typecast(md.digest(), 'uint8');
    h = lower(reshape(dec2hex(bytes, 2).', 1, []));
end
