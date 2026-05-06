function fk = compute_skeleton_fk(simOut, model_workspace_struct, opts)
%COMPUTE_SKELETON_FK  Sensor-anchored forward-kinematic validator for the golf-model skeleton.
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT) reconstructs every body-joint world
%   position (HIP -> SPINE -> TORSO -> HUB -> {LScap,RScap} -> {LS,RS} ->
%   {LE,RE} -> {LW,RW}) by combining the **logged body-frame
%   Rotation_Transform sensors** with calibrated, pose-invariant segment
%   translation offsets in each parent body frame.  It then compares the
%   reconstructed wrist positions against the directly-logged wrist
%   positions and reports the residuals.
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT, MODEL_WS_STRUCT) accepts the inputs
%   MAT struct (kept for backward compatibility; the chain itself does not
%   read segment lengths from there since the calibrated offsets already
%   encode the model geometry).
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT, MODEL_WS_STRUCT, OPTS) accepts:
%     .frame   index of the simulation frame to reconstruct (default 1, t=0)
%     .verbose (default false) — prints per-joint residuals.
%     .force_angle_chain (default false) — disables sensor-anchored mode
%               and forces the legacy angle-only chain (kept for
%               diagnostics; degrades to ~1.4 m wrist residual).
%
%   Why this exists.  The dataset_generator's SimscapeLogType='all'
%   workaround publishes every body landmark plus a 3x3 Rotation_Transform
%   for every relevant body in CombinedSignalBus.  The primary skeleton
%   extractor reads body-landmark global positions directly, but having
%   a forward-kinematic validator that reconstructs the chain from
%   parent-frame offsets is useful for (a) catching model regressions
%   where a Rigid Transform inside the model gets perturbed and (b)
%   filling missing distal joints when only proximal Transform Sensors
%   are logged.
%
%   How the chain is built (sensor-anchored, the default and accurate
%   path):
%       hip      = (HipPositionX/Y/Z)                      % logged
%       spine    = SpineLogs.GlobalPosition                % logged
%       torso    = spine + R_spine * SPINE_TO_TORSO        % calibrated offset
%       hub      = torso + R_torso * TORSO_TO_HUB
%       LScap    = hub                                     % co-located
%       RScap    = hub                                     % co-located
%       LS       = LScap + R_LScap * LSCAP_TO_LS           % [0 0 -0.254]
%       RS       = RScap + R_RScap * RSCAP_TO_RS           % [0 0 +0.254]
%       LE       = LS    + R_LS    * LS_TO_LE_BODY         % calibrated
%       RE       = RS    + R_RS    * RS_TO_RE_BODY
%       LW       = LE    + R_LF    * LF_TO_LW_BODY         % LF=Left Forearm sensor
%       RW       = RE    + R_RF    * RF_TO_RW_BODY
%
%   The constant offsets above are calibrated once against the Impact
%   pose (see CALIBRATED_OFFSETS_IMPACT.m at the bottom of this file)
%   and verified pose-invariant against TopOfBackswing.  They live in
%   each parent body frame, so each accumulated parent rotation is
%   provided directly by the model's own Transform Sensor — no
%   convention guessing required.
%
%   Status (2026-05-06).  Sensor-anchored mode produces sub-millimetre
%   wrist residuals against the logged hand positions (issue #4079).
%   The legacy angle-only path is kept behind .force_angle_chain for
%   diagnostics; it remains structurally limited because the model has
%   numerous fixed Rigid Transform blocks between joint primitives that
%   are not recoverable from joint angles alone.
%
%   See also: LOAD_IMPACT_STARTING_POSITION,
%             EXTRACTSIMSCAPEDATARECURSIVE.

    arguments
        simOut
        model_workspace_struct (1,1) struct = struct()
        opts (1,1) struct = struct()
    end

    if ~isfield(opts, 'frame');             opts.frame             = 1;     end
    if ~isfield(opts, 'verbose');           opts.verbose           = false; end
    if ~isfield(opts, 'force_angle_chain'); opts.force_angle_chain = false; end

    csb = local_safe_get(simOut, 'CombinedSignalBus');
    if isempty(csb)
        error('compute_skeleton_fk:noCSB', 'simOut has no CombinedSignalBus.');
    end

    % Pull whatever segment-length scalars are available; used only by the
    % legacy angle chain and surfaced in fk.segment_lengths for callers
    % that want them.
    L = local_resolve_lengths(model_workspace_struct);

    fk = struct();
    fk.frame = opts.frame;
    fk.mode  = 'sensor_anchored';

    fk.hip = local_xyz_scalar(csb, opts.frame, ...
        {'AngularKinematicsLogs','HipPositionX'}, ...
        {'AngularKinematicsLogs','HipPositionY'}, ...
        {'AngularKinematicsLogs','HipPositionZ'});

    % --- Try the sensor-anchored chain first. -------------------------
    sensor_ok = ~opts.force_angle_chain;
    if sensor_ok
        try
            fk = local_chain_from_sensors(fk, csb, opts.frame);
        catch ME
            sensor_ok = false;
            fk.mode   = 'angle_fallback';
            fk.sensor_chain_error = ME.message;
        end
    end

    % --- Fall back to the legacy angle-only chain when sensors absent.
    if ~sensor_ok || opts.force_angle_chain
        fk.mode = 'angle_fallback';
        Q = local_resolve_angles(csb, opts.frame);
        fk.angles_used = Q;
        fk = local_chain_from_angles(fk, L, Q);
    end

    % --- Validation against logged wrist positions. -------------------
    lw_logged = local_xyz_vec3(csb, opts.frame, {'LWLogs','LHGlobalPosition'});
    rw_logged = local_xyz_vec3(csb, opts.frame, {'RWLogs','RHGlobalPosition'});
    fk.lw_logged = lw_logged;
    fk.rw_logged = rw_logged;
    fk.lw_residual_mm = 1000 * norm(fk.lw - lw_logged);
    fk.rw_residual_mm = 1000 * norm(fk.rw - rw_logged);
    if isfield(fk, 'le') && all(~isnan(fk.le))
        le_logged = local_xyz_vec3(csb, opts.frame, {'LFLogs','GlobalPosition'});
        re_logged = local_xyz_vec3(csb, opts.frame, {'RFLogs','GlobalPosition'});
        fk.le_residual_mm = 1000 * norm(fk.le - le_logged);
        fk.re_residual_mm = 1000 * norm(fk.re - re_logged);
    end

    fk.segment_lengths = L;
    fk.calibrated_offsets = local_calibrated_offsets();

    if opts.verbose
        fprintf('[compute_skeleton_fk] mode = %s\n', fk.mode);
        if isfield(fk, 'le_residual_mm')
            fprintf('[compute_skeleton_fk] elbow residuals: LE %.2f mm, RE %.2f mm\n', ...
                    fk.le_residual_mm, fk.re_residual_mm);
        end
        fprintf('[compute_skeleton_fk] wrist residuals: LW %.2f mm, RW %.2f mm\n', ...
                fk.lw_residual_mm, fk.rw_residual_mm);
    end
end

%% =====================================================================
function fk = local_chain_from_sensors(fk, csb, idx)
%LOCAL_CHAIN_FROM_SENSORS  Build the chain using logged Rotation_Transform
%   sensors and calibrated parent-body-frame translation offsets.
    OFF = local_calibrated_offsets();

    fk.spine = local_xyz_vec3(csb, idx, {'SpineLogs','GlobalPosition'});

    R_spine = local_rotmat3(csb, idx, {'SpineLogs','Rotation_Transform'});
    fk.torso = fk.spine + (R_spine * OFF.SPINE_TO_TORSO_BODY).';

    R_torso = local_rotmat3(csb, idx, {'TorsoLogs','Rotation_Transform'});
    fk.hub  = fk.torso + (R_torso * OFF.TORSO_TO_HUB_BODY).';

    % Scapulas are co-located with HUB by design (calibration shows zero
    % offset to within 1e-4 m in either pose).
    fk.lscap = fk.hub;
    fk.rscap = fk.hub;

    R_LScap = local_rotmat3(csb, idx, {'LScapLogs','Rotation_Transform'});
    R_RScap = local_rotmat3(csb, idx, {'RScapLogs','Rotation_Transform'});
    fk.ls = fk.lscap + (R_LScap * OFF.LSCAP_TO_LS_BODY).';
    fk.rs = fk.rscap + (R_RScap * OFF.RSCAP_TO_RS_BODY).';

    R_LS = local_rotmat3(csb, idx, {'LSLogs','Rotation_Transform'});
    R_RS = local_rotmat3(csb, idx, {'RSLogs','Rotation_Transform'});
    fk.le = fk.ls + (R_LS * OFF.LS_TO_LE_BODY).';
    fk.re = fk.rs + (R_RS * OFF.RS_TO_RE_BODY).';

    % NOTE: the forearm body's reference frame origin (LFLogs.GlobalPosition)
    % is NOT the elbow joint centre — it sits offset by ~17 cm along +z in
    % the LS body frame (calibration).  We therefore use R_LF (the
    % forearm's own Rotation_Transform sensor) to add the LF-origin -> LW
    % offset, anchored at the LF GlobalPosition itself rather than at the
    % elbow.  This matches the way the model actually publishes the wrist.
    le_lf = local_xyz_vec3(csb, idx, {'LFLogs','GlobalPosition'});
    re_rf = local_xyz_vec3(csb, idx, {'RFLogs','GlobalPosition'});
    R_LF  = local_rotmat3(csb, idx, {'LFLogs','Rotation_Transform'});
    R_RF  = local_rotmat3(csb, idx, {'RFLogs','Rotation_Transform'});
    fk.lw = le_lf + (R_LF * OFF.LF_TO_LW_BODY).';
    fk.rw = re_rf + (R_RF * OFF.RF_TO_RW_BODY).';
end

%% =====================================================================
function fk = local_chain_from_angles(fk, L, Q)
%LOCAL_CHAIN_FROM_ANGLES  Legacy angle-only chain.  Kept for diagnostics
%   on older sims that don't log Transform Sensors.  Joint angles in this
%   model are stored in DEGREES; convert via deg2rad.  Joint primitive
%   composition order matches Simscape Multibody library conventions:
%       Bushing/Gimbal  -> Rx * Ry * Rz   (intrinsic XYZ)
%       Universal        -> Rx * Ry        (intrinsic XY)
%       Revolute         -> Rz             (single Z primitive)
%   The fixed Rigid Transforms inside the model are not captured here, so
%   this branch typically retains decimetre-scale residuals; the
%   sensor-anchored branch above is the accurate path.
    spine_R = rotXYZ_deg(Q.hipX, Q.hipY, Q.hipZ);
    fk.spine = fk.hip + (spine_R * [0; 0; L.UpperTorsoLength / 2]).';

    torso_R = spine_R * rotXYZ_deg(Q.spineX, Q.spineY, Q.torsoZ);
    fk.torso = fk.spine + (torso_R * [0; 0; L.UpperTorsoLength / 2]).';
    fk.hub   = fk.torso + (torso_R * [0; 0; L.UpperTorsoLength / 2]).';

    fk.lscap = fk.hub;
    fk.rscap = fk.hub;

    Lscap_R  = torso_R  * rotXY_deg(Q.LScapX, Q.LScapY);
    Rscap_R  = torso_R  * rotXY_deg(Q.RScapX, Q.RScapY);
    fk.ls = fk.lscap + (Lscap_R * [0; 0; -L.HubtoSLength]).';
    fk.rs = fk.rscap + (Rscap_R * [0; 0;  L.HubtoSLength]).';

    Lupper_R = Lscap_R * rotXYZ_deg(Q.LSx, Q.LSy, Q.LSz);
    Rupper_R = Rscap_R * rotXYZ_deg(Q.RSx, Q.RSy, Q.RSz);
    fk.le = fk.ls + (Lupper_R * [0; 0; -L.LeftUpperArmLength]).';
    fk.re = fk.rs + (Rupper_R * [0; 0; -L.RightUpperArmLength]).';

    Lfore_R = Lupper_R * rotZ_deg(Q.LE);
    Rfore_R = Rupper_R * rotZ_deg(Q.RE);
    fk.lw = fk.le + (Lfore_R * [0; 0; -(L.LowerArmLength + L.LeftWristStandoffLength)]).';
    fk.rw = fk.re + (Rfore_R * [0; 0; -(L.LowerArmLength + L.RightWristStandoffLength)]).';
end

%% =====================================================================
function OFF = local_calibrated_offsets()
%LOCAL_CALIBRATED_OFFSETS  Pose-invariant segment translation offsets in
%   each parent body frame, calibrated against the impact pose and
%   verified at top-of-backswing (issue #4079).  Units: metres.
    OFF = struct( ...
        'SPINE_TO_TORSO_BODY', [ 0.0000;  0.0000;  0.0610], ...
        'TORSO_TO_HUB_BODY',   [ 0.0000;  0.0508;  0.3048], ...
        'LSCAP_TO_LS_BODY',    [ 0.0000;  0.0000; -0.2540], ...
        'RSCAP_TO_RS_BODY',    [ 0.0000;  0.0000;  0.2540], ...
        'LS_TO_LE_BODY',       [ 0.3408;  0.0000;  0.1741], ...
        'RS_TO_RE_BODY',       [ 0.3528;  0.0000;  0.1712], ...
        'LF_TO_LW_BODY',       [-0.0195;  0.0022;  0.1940], ...
        'RF_TO_RW_BODY',       [ 0.0113; -0.0039;  0.2002]);
end

%% =====================================================================
function L = local_resolve_lengths(ws)
%LOCAL_RESOLVE_LENGTHS  Pick the segment lengths used by the legacy
%   angle chain (sensor-anchored mode does not need these).
    L = local_read_model_workspace_lengths('GolfSwing3D_Kinetic');
    fns = fieldnames(ws);
    for k = 1:numel(fns)
        if isnumeric(ws.(fns{k})) && isscalar(ws.(fns{k})) && isfield(L, fns{k})
            L.(fns{k}) = double(ws.(fns{k}));
        end
    end
    defaults = struct( ...
        'UpperTorsoLength',         0.3048, ...
        'HubtoSLength',             0.2540, ...
        'LeftShoulderWidth',        0.2540, ...
        'RightShoulderWidth',       0.2540, ...
        'LeftUpperArmLength',       0.3048, ...
        'RightUpperArmLength',      0.3048, ...
        'LowerArmLength',           0.3556, ...
        'LeftWristStandoffLength',  0.0254, ...
        'RightWristStandoffLength', 0.0254);
    fns = fieldnames(defaults);
    for k = 1:numel(fns)
        if ~isfield(L, fns{k}) || isnan(L.(fns{k}))
            L.(fns{k}) = defaults.(fns{k});
        end
    end
end

%% =====================================================================
function L = local_read_model_workspace_lengths(model_name)
%LOCAL_READ_MODEL_WORKSPACE_LENGTHS  Pull segment-length scalars from the
%   live model workspace.  Returns NaN for any name that doesn't exist.
    L = struct( ...
        'UpperTorsoLength',         NaN, ...
        'HubtoSLength',             NaN, ...
        'LeftShoulderWidth',        NaN, ...
        'RightShoulderWidth',       NaN, ...
        'LeftUpperArmLength',       NaN, ...
        'RightUpperArmLength',      NaN, ...
        'LowerArmLength',           NaN, ...
        'LeftWristStandoffLength',  NaN, ...
        'RightWristStandoffLength', NaN);
    if ~bdIsLoaded(model_name)
        return;
    end
    try
        mws = get_param(model_name, 'ModelWorkspace');
    catch
        return;
    end
    fns = fieldnames(L);
    INCHES_TO_M = 0.0254;
    for k = 1:numel(fns)
        try
            if mws.hasVariable(fns{k})
                v = mws.getVariable(fns{k});
                if isa(v, 'Simulink.Parameter'); v = v.Value; end
                if isnumeric(v) && isscalar(v)
                    L.(fns{k}) = double(v) * INCHES_TO_M;
                end
            end
        catch
        end
    end
end

%% =====================================================================
function Q = local_resolve_angles(csb, idx)
%LOCAL_RESOLVE_ANGLES  Pull the joint angles needed by the legacy chain.
%   All values returned in DEGREES (model native units).
    Q = struct();
    Q.hipX   = local_scalar(csb, {'AngularKinematicsLogs','HipAngularPositionX'}, idx);
    Q.hipY   = local_scalar(csb, {'AngularKinematicsLogs','HipAngularPositionY'}, idx);
    Q.hipZ   = local_scalar(csb, {'AngularKinematicsLogs','HipAngularPositionZ'}, idx);
    Q.spineX = local_scalar(csb, {'AngularKinematicsLogs','SpineAngularPositionX'}, idx);
    Q.spineY = local_scalar(csb, {'AngularKinematicsLogs','SpineAngularPositionY'}, idx);
    Q.torsoZ = local_scalar(csb, {'AngularKinematicsLogs','TorsoAngularPosition'}, idx);
    Q.LScapX = local_scalar(csb, {'AngularKinematicsLogs','LScapAngularPositionX'}, idx);
    Q.LScapY = local_scalar(csb, {'AngularKinematicsLogs','LScapAngularPositionY'}, idx);
    Q.RScapX = local_scalar(csb, {'AngularKinematicsLogs','RScapAngularPositionX'}, idx);
    Q.RScapY = local_scalar(csb, {'AngularKinematicsLogs','RScapAngularPositionY'}, idx);
    Q.LSx    = local_scalar(csb, {'AngularKinematicsLogs','LSAngularPositionX'}, idx);
    Q.LSy    = local_scalar(csb, {'AngularKinematicsLogs','LSAngularPositionY'}, idx);
    Q.LSz    = local_scalar(csb, {'AngularKinematicsLogs','LSAngularPositionZ'}, idx);
    Q.RSx    = local_scalar(csb, {'AngularKinematicsLogs','RSAngularPositionX'}, idx);
    Q.RSy    = local_scalar(csb, {'AngularKinematicsLogs','RSAngularPositionY'}, idx);
    Q.RSz    = local_scalar(csb, {'AngularKinematicsLogs','RSAngularPositionZ'}, idx);
    Q.LE     = local_scalar(csb, {'AngularKinematicsLogs','LEAngularPosition'}, idx);
    Q.RE     = local_scalar(csb, {'AngularKinematicsLogs','REAngularPosition'}, idx);
end

%% =====================================================================
function R = rotXYZ_deg(rx_deg, ry_deg, rz_deg)
%ROTXYZ_DEG  R = Rx(rx) * Ry(ry) * Rz(rz), inputs in degrees.
%   This is the Simscape Multibody Bushing/Gimbal joint primitive
%   composition order (X then Y then Z), intrinsic.
    if isnan(rx_deg); rx_deg = 0; end
    if isnan(ry_deg); ry_deg = 0; end
    if isnan(rz_deg); rz_deg = 0; end
    R = rotX_deg(rx_deg) * rotY_deg(ry_deg) * rotZ_deg(rz_deg);
end

%% =====================================================================
function R = rotXY_deg(rx_deg, ry_deg)
%ROTXY_DEG  R = Rx(rx) * Ry(ry), inputs in degrees.  Universal joint.
    if isnan(rx_deg); rx_deg = 0; end
    if isnan(ry_deg); ry_deg = 0; end
    R = rotX_deg(rx_deg) * rotY_deg(ry_deg);
end

%% =====================================================================
function R = rotX_deg(deg)
    if isnan(deg); deg = 0; end
    c = cosd(deg); s = sind(deg);
    R = [1, 0, 0; 0, c, -s; 0, s, c];
end

%% =====================================================================
function R = rotY_deg(deg)
    if isnan(deg); deg = 0; end
    c = cosd(deg); s = sind(deg);
    R = [c, 0, s; 0, 1, 0; -s, 0, c];
end

%% =====================================================================
function R = rotZ_deg(deg)
    if isnan(deg); deg = 0; end
    c = cosd(deg); s = sind(deg);
    R = [c, -s, 0; s, c, 0; 0, 0, 1];
end

%% =====================================================================
function val = local_scalar(csb, chain, idx)
    val = NaN;
    try
        s = csb;
        for k = 1:numel(chain)
            s = s.(chain{k});
        end
        d = s.Data;
        val = double(d(idx));
    catch
    end
end

%% =====================================================================
function v = local_xyz_scalar(csb, idx, fx, fy, fz)
    v = nan(1, 3);
    try
        v = [local_scalar(csb, fx, idx), ...
             local_scalar(csb, fy, idx), ...
             local_scalar(csb, fz, idx)];
    catch
    end
end

%% =====================================================================
function v = local_xyz_vec3(csb, idx, chain)
    v = nan(1, 3);
    try
        s = csb;
        for k = 1:numel(chain); s = s.(chain{k}); end
        d = double(s.Data);
        v = reshape(d(idx, :), 1, []);
        v = v(1:3);
    catch
    end
end

%% =====================================================================
function R = local_rotmat3(csb, idx, chain)
%LOCAL_ROTMAT3  Pull a 3x3 rotation matrix at frame IDX from
%   CSB.<chain>.Data.  Handles the [3,3,T] storage Simscape uses for
%   Rotation_Transform sensor outputs as well as [T,3,3] alternatives
%   and the trailing-singleton [3,3] shape (single-frame tests).
%   Errors are propagated so the caller can fall back to the legacy
%   angle chain.
    s = csb;
    for k = 1:numel(chain); s = s.(chain{k}); end
    d = double(s.Data);
    sz = size(d);
    if numel(sz) == 2 && sz(1) == 3 && sz(2) == 3
        % Single 3x3 with the trailing singleton stripped by MATLAB.
        R = d;
    elseif numel(sz) == 3 && sz(1) == 3 && sz(2) == 3
        R = d(:, :, idx);
    elseif numel(sz) == 3 && sz(2) == 3 && sz(3) == 3
        R = squeeze(d(idx, :, :));
    else
        error('compute_skeleton_fk:badRotShape', ...
              'Unexpected Rotation_Transform shape %s', mat2str(sz));
    end
end

%% =====================================================================
function v = local_safe_get(simOut, name)
    v = [];
    try
        if isprop(simOut, name) || isfield(simOut, name)
            v = simOut.(name);
        end
    catch
    end
end
