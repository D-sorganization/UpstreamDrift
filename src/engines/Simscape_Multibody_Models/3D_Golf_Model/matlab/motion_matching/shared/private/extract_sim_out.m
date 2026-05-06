function sim_out = extract_sim_out(simOut, joint_names, opts)
%EXTRACT_SIM_OUT  Convert raw Simulink.SimulationOutput → canonical sim_out.
%
%   SIM_OUT = EXTRACT_SIM_OUT(SIMOUT, JOINT_NAMES, OPTS) post-processes a
%   raw Simulink.SimulationOutput into the canonical struct documented in
%   simulate_with_coefficients.m.
%
%   Strategy:
%     1. Resolve a master timegrid from the most reliable signal we can find.
%     2. Resample / index every other time-varying field onto that grid.
%     3. Fill any field we cannot find with NaN of the correct shape so the
%        struct schema is stable across model revisions.
%
%   Field name mapping (CombinedSignalBus / logsout):
%     q          ← <joint>Position    (per joint)
%     qd, omega  ← <joint>Velocity
%     qdd        ← <joint>Acceleration
%     tau        ← <joint>Torque
%     r_butt     ← MidpointPosition or HandPosition
%     r_clubhead ← CHGlobalPosition or ClubheadPosition
%     q_club     ← ClubQuat or ClubOrientation
%     v_clubhead ← CHGlobalVelocity or ClubheadVelocity
%     omega_club ← ClubAngularVelocity
%
%   Preconditions:
%     - SIMOUT is a Simulink.SimulationOutput (or struct with similar shape).
%     - JOINT_NAMES is a non-empty string array.
%
%   Postconditions:
%     - sim_out.time(1) == 0 and is monotonic non-decreasing.
%     - All (N,*) fields share N rows.
%     - sim_out.solver_status ∈ {"success","warning","failed"}.

    arguments
        simOut
        joint_names (1,:) string {mustBeNonempty}
        opts (1,1) struct
    end

    n_joints = numel(joint_names);

    % --- 1. Solver status ---------------------------------------------------
    [solver_status, status_message] = local_resolve_status(simOut);

    % --- 2. Resolve master timegrid ----------------------------------------
    time = local_resolve_time(simOut, opts);
    if isempty(time)
        % Build an analytic timegrid from opts as a last resort
        dt = 1.0 / double(opts.sample_rate);
        time = (0:dt:double(opts.simulation_time))';
    end
    N = numel(time);

    % --- 3. Allocate canonical struct with NaN defaults --------------------
    sim_out = struct();
    sim_out.time         = time(:);
    sim_out.q            = nan(N, n_joints);
    sim_out.qd           = nan(N, n_joints);
    sim_out.qdd          = nan(N, n_joints);
    sim_out.tau          = nan(N, n_joints);
    sim_out.omega        = nan(N, n_joints);
    sim_out.r_butt       = nan(N, 3);
    sim_out.r_clubhead   = nan(N, 3);
    sim_out.q_club       = nan(N, 4);
    sim_out.v_clubhead   = nan(N, 3);
    sim_out.omega_club   = nan(N, 3);
    sim_out.joint_names  = joint_names;
    sim_out.solver_status = string(solver_status);
    sim_out.status_message = string(status_message);

    if solver_status == "failed"
        return;  % leave NaNs; caller decides
    end

    % --- 4. Populate joint-indexed signals ---------------------------------
    suffix_pos = ["Position", "Pos", "Angle", "q"];
    suffix_vel = ["Velocity", "Vel", "Omega", "qd"];
    suffix_acc = ["Acceleration", "Accel", "qdd"];
    suffix_tau = ["Torque", "Tau", "tau"];

    for j = 1:n_joints
        jname = joint_names(j);
        sim_out.q(:, j)   = local_pull_signal(simOut, jname, suffix_pos, time, 1);
        sim_out.qd(:, j)  = local_pull_signal(simOut, jname, suffix_vel, time, 1);
        sim_out.qdd(:, j) = local_pull_signal(simOut, jname, suffix_acc, time, 1);
        sim_out.tau(:, j) = local_pull_signal(simOut, jname, suffix_tau, time, 1);
    end
    sim_out.omega = sim_out.qd;  % alias for clarity

    % --- 5. Populate club kinematics ---------------------------------------
    sim_out.r_butt     = local_pull_named(simOut, ["MidpointPosition","ButtPosition","HandPosition","r_butt"], time, 3);
    sim_out.r_clubhead = local_pull_named(simOut, ["CHGlobalPosition","ClubheadPosition","ClubheadPos","r_clubhead"], time, 3);
    sim_out.q_club     = local_pull_named(simOut, ["ClubQuat","ClubOrientation","q_club"], time, 4);
    sim_out.v_clubhead = local_pull_named(simOut, ["CHGlobalVelocity","ClubheadVelocity","v_clubhead"], time, 3);
    sim_out.omega_club = local_pull_named(simOut, ["ClubAngularVelocity","ClubOmega","omega_club"], time, 3);

    % --- 6. Postconditions -------------------------------------------------
    assert(abs(sim_out.time(1)) < 1e-9, ...
        "extract_sim_out:timeStart", ...
        "Postcondition: sim_out.time(1) must be 0 (got %g)", sim_out.time(1));
    assert(all(diff(sim_out.time) >= -eps(max(abs(sim_out.time)))), ...
        "extract_sim_out:timeMonotonic", ...
        "Postcondition: sim_out.time must be monotonic non-decreasing");
    assert(numel(unique([size(sim_out.q,1), size(sim_out.qd,1), ...
                         size(sim_out.qdd,1), size(sim_out.tau,1), ...
                         size(sim_out.r_butt,1), size(sim_out.r_clubhead,1), ...
                         size(sim_out.q_club,1), size(sim_out.v_clubhead,1), ...
                         numel(sim_out.time)])) == 1, ...
        "extract_sim_out:rowMismatch", ...
        "Postcondition: all time-indexed fields must share N rows");
    assert(ismember(sim_out.solver_status, ["success","warning","failed"]), ...
        "extract_sim_out:badStatus", ...
        "Postcondition: solver_status must be one of {success,warning,failed}");
end

%% ----------------------------------------------------------------------
function [status, message] = local_resolve_status(simOut)
    status = "success";
    message = "";
    try
        if isprop(simOut, 'SimulationMetadata') || isfield(simOut, 'SimulationMetadata')
            md = simOut.SimulationMetadata;
            if isfield(md, 'ExecutionInfo') || isprop(md, 'ExecutionInfo')
                ex = md.ExecutionInfo;
                if isfield(ex, 'StopEvent') || isprop(ex, 'StopEvent')
                    if string(ex.StopEvent) ~= "CompletedNormally"
                        status = "failed";
                        if isfield(ex, 'ErrorDiagnostic') && ~isempty(ex.ErrorDiagnostic)
                            message = string(ex.ErrorDiagnostic.message);
                        end
                    end
                end
            end
        end
        if isprop(simOut, 'ErrorMessage') && ~isempty(simOut.ErrorMessage)
            status = "failed";
            message = string(simOut.ErrorMessage);
        end
    catch
        % Conservative: if we cannot read metadata at all assume warning
        status = "warning";
        message = "could not read SimulationMetadata";
    end
end

%% ----------------------------------------------------------------------
function t = local_resolve_time(simOut, opts) %#ok<INUSD>
    t = [];
    candidates = {'tout', 'time'};
    for k = 1:numel(candidates)
        try
            if isprop(simOut, candidates{k}) || isfield(simOut, candidates{k})
                v = simOut.(candidates{k});
                if isnumeric(v) && ~isempty(v)
                    t = double(v(:));
                    return;
                end
            end
        catch
        end
    end
    % Try logsout first signal
    try
        if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
            ls = simOut.logsout;
            if numElements(ls) >= 1
                el = ls{1};
                if isprop(el, 'Values') && isprop(el.Values, 'Time')
                    t = double(el.Values.Time(:));
                    return;
                end
            end
        end
    catch
    end
end

%% ----------------------------------------------------------------------
function vec = local_pull_signal(simOut, base_name, suffixes, time, n_cols)
%LOCAL_PULL_SIGNAL  Find a logsout / bus signal whose name is base+suffix.
    N = numel(time);
    if n_cols == 1
        vec = nan(N, 1);
    else
        vec = nan(N, n_cols);
    end
    for s = 1:numel(suffixes)
        name = char(base_name) + string(suffixes(s));
        v = local_lookup(simOut, name, time);
        if ~isempty(v)
            vec = local_to_columns(v, time, n_cols);
            return;
        end
    end
end

%% ----------------------------------------------------------------------
function vec = local_pull_named(simOut, names, time, n_cols)
    N = numel(time);
    vec = nan(N, n_cols);
    for k = 1:numel(names)
        v = local_lookup(simOut, char(names(k)), time);
        if ~isempty(v)
            vec = local_to_columns(v, time, n_cols);
            return;
        end
    end
end

%% ----------------------------------------------------------------------
function v = local_lookup(simOut, name, time) %#ok<INUSD>
%LOCAL_LOOKUP  Try CombinedSignalBus, logsout, and direct properties.
    v = [];
    name = char(name);

    % CombinedSignalBus.<name>
    try
        if (isprop(simOut, 'CombinedSignalBus') || isfield(simOut, 'CombinedSignalBus'))
            bus = simOut.CombinedSignalBus;
            v = local_dig(bus, name);
            if ~isempty(v), return; end
        end
    catch
    end

    % logsout get by name
    try
        if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
            ls = simOut.logsout;
            try
                el = ls.getElement(name);
                if ~isempty(el) && isprop(el, 'Values')
                    v = local_extract_values(el.Values);
                    if ~isempty(v), return; end
                end
            catch
            end
        end
    catch
    end

    % Direct property / field on simOut
    try
        if isprop(simOut, name) || isfield(simOut, name)
            v = simOut.(name);
            if isstruct(v)
                v = local_extract_values(v);
            end
        end
    catch
    end
end

%% ----------------------------------------------------------------------
function v = local_dig(bus, name)
%LOCAL_DIG  Recursive search through nested struct buses for a field.
    v = [];
    if ~isstruct(bus), return; end
    if isfield(bus, name)
        v = local_extract_values(bus.(name));
        return;
    end
    f = fieldnames(bus);
    for i = 1:numel(f)
        sub = bus.(f{i});
        if isstruct(sub)
            v = local_dig(sub, name);
            if ~isempty(v), return; end
        end
    end
end

%% ----------------------------------------------------------------------
function v = local_extract_values(x)
%LOCAL_EXTRACT_VALUES  Pull numeric data from timeseries / struct-with-Data.
    v = [];
    try
        if isnumeric(x)
            v = double(x);
        elseif isa(x, 'timeseries')
            v = double(x.Data);
        elseif isstruct(x)
            if isfield(x, 'Data')
                v = double(x.Data);
            elseif isfield(x, 'signals') && isfield(x.signals, 'values')
                v = double(x.signals.values);
            end
        end
    catch
        v = [];
    end
    if ~isempty(v) && ndims(v) > 2  % squeeze trailing singletons
        v = squeeze(v);
    end
end

%% ----------------------------------------------------------------------
function out = local_to_columns(v, time, n_cols)
%LOCAL_TO_COLUMNS  Coerce a raw signal sample to (N, n_cols) matching time.
    N = numel(time);
    if isempty(v)
        out = nan(N, n_cols);
        return;
    end
    if isvector(v) && n_cols == 1
        out = local_resample(v(:), N);
    else
        % shape (N, n_cols) or (n_cols, N)
        sz = size(v);
        if sz(1) == n_cols && sz(2) ~= n_cols
            v = v.';
        end
        if size(v, 2) > n_cols
            v = v(:, 1:n_cols);
        elseif size(v, 2) < n_cols
            pad = nan(size(v,1), n_cols - size(v,2));
            v = [v, pad];
        end
        out = nan(N, n_cols);
        for c = 1:n_cols
            out(:, c) = local_resample(v(:, c), N);
        end
    end
end

%% ----------------------------------------------------------------------
function y = local_resample(x, N)
%LOCAL_RESAMPLE  Resample a vector to length N by linear interp on its index.
    x = x(:);
    n = numel(x);
    if n == N
        y = x;
    elseif n == 0
        y = nan(N, 1);
    else
        ix = linspace(1, n, N).';
        y = interp1((1:n).', x, ix, 'linear', 'extrap');
    end
end
