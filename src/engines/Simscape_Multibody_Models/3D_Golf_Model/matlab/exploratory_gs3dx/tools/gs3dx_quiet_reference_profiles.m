function ref = gs3dx_quiet_reference_profiles(targets, end_time)
%GS3DX_QUIET_REFERENCE_PROFILES  Hold reference profiles for quiet stance (#10979).
%
%   REF = GS3DX_QUIET_REFERENCE_PROFILES(TARGETS, END_TIME) maps initial kinematics
%   targets (from GS3DX_INITIAL_TARGET_VALUES) into constant leg and upper-body
%   tracking reference profiles spanning [0, END_TIME].
%
%   Outputs:
%     REF.workspace_values: LegReferenceTime [0, END_TIME] (1x2), LegReferenceAngle
%       (12x2 deg, order [LHip(3); LKnee(1); LAnkle(2); RHip(3); RKnee(1); RAnkle(2)]),
%       LegReferenceRate (12x2 zeros), UpperBodyTrackTime [0, END_TIME], for all 12
%       joints from GS3DX_UPPER_BODY_JOINTS <prefix>TrackAngle (numel(axes)x2),
%       <prefix>TrackRate (zeros), <prefix>TrackTorque (zeros in N*m, explicitly NOT
%       gravity compensated), and NeckReference (2x2 rad, converted from literal Rx.q/Ry.q in deg).
%     REF.qualification: 'CONSTANT_REFERENCE_PROFILE_MAPPING_ONLY'
%     REF.balance_reference_policy: 'REQUIRES_SEPARATE_NATIVE_COM_AND_FOOT_CONFIGURATION'
%
%   Caller Responsibilities & Non-Claims:
%     Caller must separately bind initial targets (Translation, Pelvis/Hip, velocities),
%     contact plane, gravity, support switches, and native COM/foot balance references.
%     Zero feedforward torque does NOT hold gravity; active closed loop is required.

    arguments
        targets
        end_time
    end

    err_id = 'gs3dx:quiet_reference';

    % 1. Validate END_TIME
    if ~isnumeric(end_time) || ~isreal(end_time) || ~isscalar(end_time) || ...
            ~isfinite(end_time) || end_time <= 0
        error(err_id, 'END_TIME must be a finite real positive scalar.');
    end
    t_end = double(end_time);

    % 2. Validate TARGETS struct and qualification
    if ~isstruct(targets) || ~isscalar(targets) || ...
            ~isfield(targets, 'qualification') || ...
            ~isfield(targets, 'workspace_values') || ...
            ~isfield(targets, 'block_position_values') || ...
            ~isstruct(targets.workspace_values) || ~isscalar(targets.workspace_values) || ...
            ~isstruct(targets.block_position_values)
        error(err_id, 'TARGETS must be a scalar struct qualified as PURE_MAPPING_ONLY.');
    end

    tag = targets.qualification;
    if ~((ischar(tag) && isrow(tag)) || (isstring(tag) && isscalar(tag))) || ...
            ismissing(string(tag)) || string(tag) ~= "PURE_MAPPING_ONLY"
        error(err_id,'TARGETS qualification must be scalar PURE_MAPPING_ONLY text.');
    end

    ws = targets.workspace_values;
    bp = targets.block_position_values;

    % 3. Extract and validate leg start positions (12 coordinates)
    leg_req = { ...
        'LHipStartPosition', [3, 1]; ...
        'LKneeStartPosition', [1, 1]; ...
        'LAnkleStartPosition', [2, 1]; ...
        'RHipStartPosition', [3, 1]; ...
        'RKneeStartPosition', [1, 1]; ...
        'RAnkleStartPosition', [2, 1] ...
    };
    for k = 1:size(leg_req, 1)
        fn = leg_req{k, 1};
        sz = leg_req{k, 2};
        if ~isfield(ws, fn)
            error(err_id, 'Missing required leg coordinate: %s', fn);
        end
        v = ws.(fn);
        if ~isnumeric(v) || ~isreal(v) || ~isequal(size(v), sz) || any(~isfinite(v(:)))
            error(err_id, 'Leg coordinate %s must be real, finite, size %s.', fn, mat2str(sz));
        end
    end

    q_legs = [ ...
        double(ws.LHipStartPosition); ...
        double(ws.LKneeStartPosition); ...
        double(ws.LAnkleStartPosition); ...
        double(ws.RHipStartPosition); ...
        double(ws.RKneeStartPosition); ...
        double(ws.RAnkleStartPosition) ...
    ];

    out_ws = struct();
    out_ws.LegReferenceTime = [0.0, t_end];
    out_ws.LegReferenceAngle = repmat(double(q_legs), 1, 2);
    out_ws.LegReferenceRate = zeros(12, 2);
    out_ws.UpperBodyTrackTime = [0.0, t_end];

    % 4. Extract and validate 12 upper-body joints via gs3dx_upper_body_joints
    spec = gs3dx_upper_body_joints();
    for k = 1:numel(spec)
        j = spec(k);
        if isempty(j.axes)
            fn = [j.prefix 'StartPosition'];
            if ~isfield(ws, fn)
                error(err_id, 'Missing upper body coordinate: %s', fn);
            end
            val = ws.(fn);
            if ~isnumeric(val) || ~isreal(val) || ~isscalar(val) || ~isfinite(val)
                error(err_id, 'Coordinate %s must be a finite real scalar.', fn);
            end
            ang = [double(val), double(val)];
            n_ax = 1;
        else
            n_ax = numel(j.axes);
            vec = zeros(n_ax, 1);
            for a = 1:n_ax
                fn = [j.prefix 'StartPosition' j.axes(a)];
                if ~isfield(ws, fn)
                    error(err_id, 'Missing upper body coordinate: %s', fn);
                end
                val = ws.(fn);
                if ~isnumeric(val) || ~isreal(val) || ~isscalar(val) || ~isfinite(val)
                    error(err_id, 'Coordinate %s must be a finite real scalar.', fn);
                end
                vec(a) = double(val);
            end
            ang = repmat(vec, 1, 2);
        end
        out_ws.([j.prefix 'TrackAngle']) = ang;
        out_ws.([j.prefix 'TrackRate']) = zeros(n_ax, 2);
        out_ws.([j.prefix 'TrackTorque']) = zeros(n_ax, 2);
    end

    % 5. Extract and validate literal Neck Joint (Rx.q, Ry.q in deg)
    if ~all(isfield(bp, {'block', 'primitive', 'value', 'unit'}))
        error(err_id, 'block_position_values missing required schema fields.');
    end
    neck_blk = "Hips and Torso Inputs/Neck Joint";
    b_names = string({bp.block});
    p_names = string({bp.primitive});
    i_rx = find(b_names == neck_blk & p_names == "Rx.q");
    i_ry = find(b_names == neck_blk & p_names == "Ry.q");

    if numel(i_rx) ~= 1 || numel(i_ry) ~= 1
        error(err_id, 'Expected unique literal bindings for Neck Joint Rx.q and Ry.q.');
    end

    entry_rx = bp(i_rx);
    entry_ry = bp(i_ry);
    rx_unit = entry_rx.unit;
    ry_unit = entry_ry.unit;
    if ~local_scalar_degree_unit(rx_unit) || ~local_scalar_degree_unit(ry_unit)
        error(err_id, 'Neck Joint literal target units must be deg.');
    end
    rx_v = entry_rx.value;
    ry_v = entry_ry.value;
    if ~isnumeric(rx_v) || ~isreal(rx_v) || ~isscalar(rx_v) || ~isfinite(rx_v) || ...
       ~isnumeric(ry_v) || ~isreal(ry_v) || ~isscalar(ry_v) || ~isfinite(ry_v)
        error(err_id, 'Neck Joint angles must be real finite scalars.');
    end
    out_ws.NeckReference = repmat(deg2rad([double(rx_v); double(ry_v)]), 1, 2);

    % 6. Assemble return struct
    ref = struct( ...
        'workspace_values', out_ws, ...
        'units', struct('LegReferenceTime','s','LegReferenceAngle','deg', ...
            'LegReferenceRate','deg/s','UpperBodyTrackTime','s', ...
            'TrackAngle','deg','TrackRate','deg/s','TrackTorque','N*m','NeckReference','rad'), ...
        'qualification', 'CONSTANT_REFERENCE_PROFILE_MAPPING_ONLY', ...
        'balance_reference_policy', 'REQUIRES_SEPARATE_NATIVE_COM_AND_FOOT_CONFIGURATION');
end

function yes = local_scalar_degree_unit(unit)
    yes = ((ischar(unit) && isrow(unit)) || (isstring(unit) && isscalar(unit)));
    if yes
        text = string(unit);
        yes = ~ismissing(text) && text == "deg";
    end
end
