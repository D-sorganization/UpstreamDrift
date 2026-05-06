function fk = compute_skeleton_fk(simOut, model_workspace_struct, opts)
%COMPUTE_SKELETON_FK  Forward-kinematic reconstruction of body-joint world positions.
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT) walks down the model's body chain
%   using the joint angles published in CombinedSignalBus.AngularKinematicsLogs
%   plus the segment lengths from the SimulationInput's modelworkspace,
%   and returns a struct with the world-frame positions of every body
%   landmark (HIP → SPINE → HUB → LS/RS → LE/RE → LW/RW).
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT, MODEL_WS_STRUCT) lets the caller
%   supply the model-workspace struct directly (handy when running from a
%   MAT file: load('3DModelInputs_Impact.mat') gives such a struct).
%
%   FK = COMPUTE_SKELETON_FK(SIMOUT, MODEL_WS_STRUCT, OPTS) accepts:
%     .frame   index of the simulation frame to reconstruct (default 1, t=0)
%     .verbose (default false) — prints per-segment chain lengths and the
%               residual error vs. the directly-logged wrist positions.
%
%   Why this exists. The dataset_generator's `SimscapeLogType='all'`
%   workaround already publishes shoulder, elbow, and wrist global
%   positions in CombinedSignalBus, so the primary skeleton extractor
%   reads them directly.  This function is the **fallback** for older
%   sims where that flag was off, and the **validator** that lets us
%   detect a sign-convention mismatch between the model and our chain
%   model by comparing the FK wrist against the logged wrist.
%
%   The chain we model (segment names verified against
%   src/model/mdl_reference/GolfSwing3D_Kinetic.mdl):
%       hip      = (HipPositionX, Y, Z)                   % logged
%       spine_R  = Rz(HipAngZ) Ry(HipAngY) Rx(HipAngX)
%       spine    = hip + spine_R * [0;0;UpperTorsoLength/2]
%       torso_R  = spine_R * Rz(TorsoAng) Ry(SpineAngY) Rx(SpineAngX)
%       hub      = spine + torso_R * [0;0;UpperTorsoLength/2]
%       LS       = hub + torso_R * [-(HubtoSLength+LeftShoulderWidth);0;0]
%       RS       = hub + torso_R * [ (HubtoSLength+RightShoulderWidth);0;0]
%       Lscap_R  = torso_R  * Ry(LScapAngY) Rx(LScapAngX)
%       Rscap_R  = torso_R  * Ry(RScapAngY) Rx(RScapAngX)
%       Lupper_R = Lscap_R  * Rz(LSAngZ) Ry(LSAngY) Rx(LSAngX)
%       Rupper_R = Rscap_R  * Rz(RSAngZ) Ry(RSAngY) Rx(RSAngX)
%       LE       = LS + Lupper_R * [0;0;-LeftUpperArmLength]
%       RE       = RS + Rupper_R * [0;0;-RightUpperArmLength]
%       Lfore_R  = Lupper_R * Rz(LEAngularPosition)
%       Rfore_R  = Rupper_R * Rz(REAngularPosition)
%       LW       = LE + Lfore_R * [0;0;-(LowerArmLength+LeftWristStandoffLength)]
%       RW       = RE + Rfore_R * [0;0;-(LowerArmLength+RightWristStandoffLength)]
%
%   Status (2026-05-06): the chain runs end-to-end and reads segment
%   lengths from the live model workspace (converting from inches to
%   metres), but the rotation order on the spine→shoulder→elbow path is
%   not yet exactly right — current LW/RW residuals against the logged
%   wrist positions are on the order of 100–1500 mm at t=0.  This is a
%   known limitation: the helper is intentionally kept as a *validator*
%   (does the chain land near the logged wrist?) and *fallback* (fill
%   missing joints when the model isn't logging them) rather than the
%   primary skeleton source.  The primary path
%   (`load_impact_starting_position` reading body landmark global
%   positions directly from `CombinedSignalBus`) is sub-mm accurate by
%   construction and should be used whenever those signals are present.
%
%   When the FK and the logged wrist disagree by more than a few cm at
%   t=0, suspect (a) a flipped axis in the rotation order versus what
%   the .slx actually uses internally, or (b) a Simscape body element
%   with a non-default RFrame/BFrame offset that isn't captured by the
%   simple "[0;0;-length]" assumption.  Update the chain in
%   `compute_skeleton_fk.m` after inspecting
%   `src/model/mdl_reference/GolfSwing3D_Kinetic.mdl` for the exact
%   joint orientation parameters.
%
%   See also: LOAD_IMPACT_STARTING_POSITION,
%             EXTRACTSIMSCAPEDATARECURSIVE.

    arguments
        simOut
        model_workspace_struct (1,1) struct = struct()
        opts (1,1) struct = struct()
    end

    if ~isfield(opts, 'frame');   opts.frame   = 1;     end
    if ~isfield(opts, 'verbose'); opts.verbose = false; end

    csb = local_safe_get(simOut, 'CombinedSignalBus');
    if isempty(csb)
        error('compute_skeleton_fk:noCSB', 'simOut has no CombinedSignalBus.');
    end

    L = local_resolve_lengths(model_workspace_struct);
    Q = local_resolve_angles(csb, opts.frame);

    fk = struct();
    fk.frame = opts.frame;

    % --- Chain reconstruction. -----------------------------------------
    fk.hip   = local_xyz_scalar(csb, opts.frame, ...
                  {'AngularKinematicsLogs','HipPositionX'}, ...
                  {'AngularKinematicsLogs','HipPositionY'}, ...
                  {'AngularKinematicsLogs','HipPositionZ'});

    spine_R = rotZYX(Q.hipZ, Q.hipY, Q.hipX);
    fk.spine = fk.hip + (spine_R * [0; 0; L.UpperTorsoLength / 2])';

    torso_R = spine_R * rotZYX(Q.torsoZ, Q.spineY, Q.spineX);
    fk.hub  = fk.spine + (torso_R * [0; 0; L.UpperTorsoLength / 2])';

    % Hub-to-shoulder offset is split: HubtoSLength gives the lateral arm
    % from the spine column, then LeftShoulderWidth / RightShoulderWidth
    % continue out along ±x to the actual shoulder joints.
    fk.ls = fk.hub + (torso_R * [-(L.HubtoSLength + L.LeftShoulderWidth);  0; 0])';
    fk.rs = fk.hub + (torso_R * [ (L.HubtoSLength + L.RightShoulderWidth); 0; 0])';

    Lscap_R  = torso_R * rotZYX(0, Q.LScapY, Q.LScapX);
    Rscap_R  = torso_R * rotZYX(0, Q.RScapY, Q.RScapX);
    Lupper_R = Lscap_R * rotZYX(Q.LSz, Q.LSy, Q.LSx);
    Rupper_R = Rscap_R * rotZYX(Q.RSz, Q.RSy, Q.RSx);

    fk.le = fk.ls + (Lupper_R * [0; 0; -L.LeftUpperArmLength])';
    fk.re = fk.rs + (Rupper_R * [0; 0; -L.RightUpperArmLength])';

    % Forearm length is shared between sides in this model
    % (`LowerArmLength`, no Left/Right variant).
    Lfore_R = Lupper_R * rotZYX(Q.LE, 0, 0);
    Rfore_R = Rupper_R * rotZYX(Q.RE, 0, 0);

    fk.lw = fk.le + (Lfore_R * [0; 0; -(L.LowerArmLength + L.LeftWristStandoffLength)])';
    fk.rw = fk.re + (Rfore_R * [0; 0; -(L.LowerArmLength + L.RightWristStandoffLength)])';

    % --- Validation against logged wrist positions. --------------------
    lw_logged = local_xyz_vec3(csb, opts.frame, {'LWLogs','LHGlobalPosition'});
    rw_logged = local_xyz_vec3(csb, opts.frame, {'RWLogs','RHGlobalPosition'});
    fk.lw_logged = lw_logged;
    fk.rw_logged = rw_logged;
    fk.lw_residual_mm = 1000 * norm(fk.lw - lw_logged);
    fk.rw_residual_mm = 1000 * norm(fk.rw - rw_logged);

    fk.segment_lengths = L;
    fk.angles_used     = Q;

    if opts.verbose
        fprintf('[compute_skeleton_fk] segment lengths used:\n');
        fns = fieldnames(L);
        for k = 1:numel(fns)
            fprintf('   %-25s = %7.4f m\n', fns{k}, L.(fns{k}));
        end
        fprintf('[compute_skeleton_fk] FK→wrist residuals: LW %.1f mm, RW %.1f mm\n', ...
                fk.lw_residual_mm, fk.rw_residual_mm);
    end
end

%% =====================================================================
function L = local_resolve_lengths(ws)
%LOCAL_RESOLVE_LENGTHS  Pick the segment lengths used by the FK chain.
%   The actual model-workspace names (verified against
%   src/model/mdl_reference/GolfSwing3D_Kinetic.mdl) are:
%       UpperTorsoLength, LowerTorsoLength
%       LeftShoulderWidth, RightShoulderWidth, HubtoSLength
%       UpperArmLength, LeftUpperArmLength, RightUpperArmLength
%       LowerArmLength       (forearm — single name for both sides)
%       LeftWristStandoffLength, RightWristStandoffLength
%   These live ONLY in the model workspace, never in the input MAT.
%   So we first try to read from the live `GolfSwing3D_Kinetic` model
%   workspace, and fall back to whatever the caller passed in `ws`.
    L = local_read_model_workspace_lengths('GolfSwing3D_Kinetic');
    fns = fieldnames(ws);
    for k = 1:numel(fns)
        if isnumeric(ws.(fns{k})) && isscalar(ws.(fns{k})) && isfield(L, fns{k})
            L.(fns{k}) = double(ws.(fns{k}));
        end
    end
    % Fill any field that's still NaN with a plausible default so the FK
    % at least returns finite numbers (the residual against logged wrists
    % will reveal which one to source-correct).
    defaults = struct( ...
        'UpperTorsoLength',         0.528, ...
        'HubtoSLength',             0.150, ...
        'LeftShoulderWidth',        0.190, ...
        'RightShoulderWidth',       0.190, ...
        'LeftUpperArmLength',       0.310, ...
        'RightUpperArmLength',      0.310, ...
        'LowerArmLength',           0.290, ...
        'LeftWristStandoffLength',  0.030, ...
        'RightWristStandoffLength', 0.030);
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
%   live model workspace.  Returns NaN for any name that doesn't exist
%   so the caller can apply defaults.
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
                % Unwrap Simulink.Parameter; live model workspaces store
                % these as parameter objects rather than plain doubles.
                if isa(v, 'Simulink.Parameter'); v = v.Value; end
                if isnumeric(v) && isscalar(v)
                    % Body dimensions in this model are authored in inches
                    % (UpperTorsoLength = 12 in, LowerArmLength = 14 in,
                    % etc.) and the .slx applies the conversion via the
                    % Simscape body-element units.  Apply it explicitly
                    % here so our metric FK chain matches the metric
                    % global-position outputs.
                    L.(fns{k}) = double(v) * INCHES_TO_M;
                end
            end
        catch
        end
    end
end

%% =====================================================================
function Q = local_resolve_angles(csb, idx)
%LOCAL_RESOLVE_ANGLES  Pull the joint angles needed by the FK chain at FRAME idx.
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
function R = rotZYX(rz, ry, rx)
%ROTZYX  ZYX intrinsic Euler rotation matrix R = Rz(rz) Ry(ry) Rx(rx).
    if isnan(rz); rz = 0; end
    if isnan(ry); ry = 0; end
    if isnan(rx); rx = 0; end
    cz = cos(rz); sz = sin(rz);
    cy = cos(ry); sy = sin(ry);
    cx = cos(rx); sx = sin(rx);
    R = [cz, -sz, 0; sz, cz, 0; 0, 0, 1] ...
      * [cy,  0, sy;  0,  1, 0; -sy, 0, cy] ...
      * [ 1,  0,  0;  0, cx, -sx; 0, sx, cx];
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
function v = local_safe_get(simOut, name)
    v = [];
    try
        if isprop(simOut, name) || isfield(simOut, name)
            v = simOut.(name);
        end
    catch
    end
end
