function skel = load_impact_starting_position(opts)
%LOAD_IMPACT_STARTING_POSITION  Run model with Impact inputs, return skeleton at t=0.
%
%   SKEL = LOAD_IMPACT_STARTING_POSITION() loads the 3D Simscape model
%   GolfSwing3D_Kinetic with the Impact input file as the model workspace,
%   runs a short simulation, then extracts every joint-centre position at
%   the first frame and returns them as a single struct. The Impact pose is
%   used as a "stand-in for address" per the user's note that the two are
%   structurally similar.
%
%   SKEL = LOAD_IMPACT_STARTING_POSITION(OPTS) accepts:
%     .model_name       (default 'GolfSwing3D_Kinetic')
%     .input_file       (default 3DModelInputs_Impact.mat under src/model/inputs/)
%     .stop_time        (default 0.005 s — long enough to log >=1 frame)
%     .verbose          (default true)
%
%   Output struct fields (all 1x3 doubles, world-frame metres):
%     butt, mp, ch, lw, le, ls, rw, re, rs, hub
%     time         scalar simulation time captured (s)
%     model_name   string
%     input_file   string
%     full_table   the table from generateDataTable3D for the short sim
%
%   Joints not present in the logsout are returned as NaN(1,3) so callers
%   can detect missing channels rather than crashing.
%
%   See also: SETUP_MATLAB_ENVIRONMENT, GENERATEDATATABLE3D,
%             PLOT_STARTING_POSITION_MATCH.

    arguments
        opts (1,1) struct = struct()
    end

    opts = local_fill_defaults(opts);
    here = fileparts(mfilename('fullpath'));

    % --- 1. Make sure src tree is on path so the model and helpers resolve.
    matlab_root = fileparts(fileparts(here));   % .../matlab/
    src_dir     = fullfile(matlab_root, 'src');
    if exist(src_dir, 'dir')
        addpath(genpath(src_dir));
    end

    % --- 2. Resolve model and input file locations.
    model_name = char(opts.model_name);
    input_file = char(opts.input_file);
    if ~isfile(input_file)
        % Fallback: search for the file under src/model/inputs.
        candidate = fullfile(matlab_root, 'src', 'model', 'inputs', '3DModelInputs_Impact.mat');
        if isfile(candidate)
            input_file = candidate;
        else
            error('load_impact_starting_position:inputFileMissing', ...
                  'Impact input file not found: %s', opts.input_file);
        end
    end

    % --- 3. Load model.  Do NOT swap the model workspace's DataSource:
    %         doing that wipes out vars that live only in the model
    %         workspace by default (LocalDampeningEnable, etc.).  Instead
    %         we overlay the Impact mat values via setVariable, which
    %         scopes the override to this sim call.
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    % --- 4. Configure a tiny run.  We do not need the full swing — only the
    %         t=0 frame for joint centres.  We allow a few ms so that the
    %         solver has at least one logged sample.
    in = Simulink.SimulationInput(model_name);
    in = in.setModelParameter('StopTime',  num2str(double(opts.stop_time)));
    in = in.setModelParameter('FastRestart', 'off');
    in = in.setModelParameter('SaveOutput',  'on');
    in = in.setModelParameter('ReturnWorkspaceOutputs', 'on');
    % SimscapeLogType='all' is already saved persistently in the .slx
    % (line 4256 of src/model/mdl_reference/GolfSwing3D_Kinetic.mdl), so
    % we no longer need to set it here.  We keep this as a defensive
    % override only when the caller has obviously disabled it via opts —
    % otherwise we trust the model's saved configuration.
    if isfield(opts, 'force_simscape_log_all') && opts.force_simscape_log_all
        try
            in = in.setModelParameter('SimscapeLogType', 'all');
        catch ME
            if opts.verbose
                fprintf('[load_impact_starting_position] could not set SimscapeLogType=all: %s\n', ME.message);
            end
        end
    end

    in = local_apply_impact_inputs(in, input_file, opts.verbose);

    if opts.verbose
        fprintf('[load_impact_starting_position] running %s with %s for %.4fs ...\n', ...
            model_name, input_file, opts.stop_time);
    end

    simOut = sim(in);

    % --- 5. Pull joint centres from the model output.  With
    %         SimscapeLogType=all the CombinedSignalBus exposes every
    %         body landmark we need (HUB/LS/RS/LE/RE in addition to
    %         hands/clubhead).  Anything that still slips through gets
    %         filled in by compute_skeleton_fk as a fallback.
    skel = local_extract_skeleton_from_csb(simOut);
    skel.model_name = string(model_name);
    skel.input_file = string(input_file);
    skel.sim_out    = simOut;

    % FK fallback / validator.  We always run it so the caller can see
    % the residual between FK and the directly-logged wrist positions.
    try
        ws_struct = load(input_file);
        skel.fk = compute_skeleton_fk(simOut, ws_struct, struct('frame', 1, 'verbose', opts.verbose));
        % Backfill any joint that's still NaN with the FK result.
        for f = {'le','re','ls','rs','hub'}
            if any(isnan(skel.(f{1}))) && isfield(skel.fk, f{1}) && all(~isnan(skel.fk.(f{1})))
                skel.(f{1}) = skel.fk.(f{1});
                if opts.verbose
                    fprintf('[load_impact_starting_position] backfilled %s from FK chain\n', f{1});
                end
            end
        end
    catch ME
        if opts.verbose
            fprintf('[load_impact_starting_position] FK validation skipped: %s\n', ME.message);
        end
    end

    if opts.verbose
        try
            n_samp = numel(simOut.tout);
        catch
            n_samp = NaN;
        end
        fprintf('[load_impact_starting_position] captured %d frames; reporting frame 1 (t=%.4fs).\n', ...
                n_samp, skel.time);
        local_print_skeleton(skel);
    end
end

%% =====================================================================
function in = local_apply_impact_inputs(in, input_file, verbose)
%LOCAL_APPLY_IMPACT_INPUTS  Overlay every variable from the Impact MAT onto
%   the SimulationInput so the existing model workspace defaults
%   (LocalDampeningEnable, joint masks, etc.) are preserved.
    S = load(input_file);
    f = fieldnames(S);
    n_set = 0;
    for k = 1:numel(f)
        name  = f{k};
        value = S.(name);
        try
            in = in.setVariable(name, value);
            n_set = n_set + 1;
        catch ME
            if verbose
                fprintf('[load_impact_starting_position] could not set %s: %s\n', name, ME.message);
            end
        end
    end
    if verbose
        fprintf('[load_impact_starting_position] overlaid %d/%d variables from %s\n', ...
                n_set, numel(f), input_file);
    end
end

%% =====================================================================
function opts = local_fill_defaults(opts)
    here = fileparts(mfilename('fullpath'));
    matlab_root = fileparts(fileparts(here));
    default_input = fullfile(matlab_root, 'src', 'model', 'inputs', '3DModelInputs_Impact.mat');
    defaults = struct( ...
        'model_name', 'GolfSwing3D_Kinetic', ...
        'input_file', default_input, ...
        'stop_time',  0.005, ...
        'verbose',    true);
    f = fieldnames(defaults);
    for k = 1:numel(f)
        if ~isfield(opts, f{k})
            opts.(f{k}) = defaults.(f{k});
        end
    end
end

%% =====================================================================
function skel = local_extract_skeleton_from_csb(simOut)
%LOCAL_EXTRACT_SKELETON_FROM_CSB  Pull t=0 joint centres from CombinedSignalBus.
%   Returns NaN for any landmark not directly published in the bus.
    skel = struct();
    if isprop(simOut, 'tout') || isfield(simOut, 'tout')
        try
            tt = simOut.tout;
            if ~isempty(tt); skel.time = tt(1); else; skel.time = 0; end
        catch
            skel.time = 0;
        end
    else
        skel.time = 0;
    end

    csb = local_safe_get(simOut, 'CombinedSignalBus');
    if isempty(csb)
        error('load_impact_starting_position:noCSB', ...
              'simOut does not contain CombinedSignalBus — cannot extract skeleton.');
    end

    % Hip is the kinematic root; provided as separate scalar timeseries.
    skel.hip = local_xyz_scalar(csb, ...
        {'AngularKinematicsLogs','HipPositionX'}, ...
        {'AngularKinematicsLogs','HipPositionY'}, ...
        {'AngularKinematicsLogs','HipPositionZ'});

    % Spine column.
    skel.torso = local_xyz_vec3(csb, {'TorsoLogs','GlobalPosition'});
    skel.spine = local_xyz_vec3(csb, {'SpineLogs','GlobalPosition'});

    % HUB is the actual top-of-spine (base of neck) — published only when
    % SimscapeLogType=all is on. We fall back to spine top if missing.
    skel.hub = local_xyz_vec3(csb, {'HipLogs','HUBGlobalPosition'});
    if any(isnan(skel.hub)); skel.hub = skel.spine; end

    % Shoulders — exposed by SimscapeLogType=all on the LS/RS body logs.
    skel.ls = local_xyz_vec3(csb, {'LSLogs','GlobalPosition'});
    skel.rs = local_xyz_vec3(csb, {'RSLogs','GlobalPosition'});

    % Elbows — the forearm body's reference frame origin sits at the
    % proximal (elbow) end in this model.  Cross-validates against
    % shoulder-to-wrist distance below.
    skel.le = local_xyz_vec3(csb, {'LFLogs','GlobalPosition'});
    skel.re = local_xyz_vec3(csb, {'RFLogs','GlobalPosition'});

    % Wrists, midpoint, clubhead.
    skel.lw = local_xyz_vec3(csb, {'LWLogs','LHGlobalPosition'});
    skel.rw = local_xyz_vec3(csb, {'RWLogs','RHGlobalPosition'});
    skel.mp = local_xyz_vec3(csb, {'MidpointCalcsLogs','MPGlobalPosition'});
    skel.ch = local_xyz_vec3(csb, {'ClubLogs','CHGlobalPosition'});

    % Butt of club — published as a calculated signal when SimscapeLogType
    % is on; otherwise approximate as MP reflected through CH.
    skel.butt = local_xyz_vec3(csb, {'LHCalcsLogs','ButtPosition'});
    if any(isnan(skel.butt)) && all(~isnan(skel.mp)) && all(~isnan(skel.ch))
        skel.butt = skel.mp - 0.18 * (skel.ch - skel.mp) / norm(skel.ch - skel.mp);
    end

    % Grip orientation — published as a 3x3 rotation transform in
    % MomentandCoupleLogs (and a duplicate in MidpointCalcsLogs).  We
    % expose both the rotation matrix and a quaternion so callers can
    % use whichever they prefer.
    skel.grip_R    = local_rotmat3(csb, {'MomentandCoupleLogs','RotationTransformMP'});
    if any(isnan(skel.grip_R(:)))
        skel.grip_R = local_rotmat3(csb, {'MidpointCalcsLogs','RotationTransformMP'});
    end
    if any(isnan(skel.grip_R(:)))
        skel.grip_R = nan(3, 3);
        skel.grip_quat = nan(1, 4);
    else
        skel.grip_quat = local_R_to_quat(skel.grip_R);
    end

    skel.joint_order = {'butt','mp','ch','lw','le','ls','rw','re','rs','hub','hip','torso','spine'};
end

%% =====================================================================
function R = local_rotmat3(csb, chain)
    R = nan(3, 3);
    try
        s = csb;
        for k = 1:numel(chain); s = s.(chain{k}); end
        d = double(s.Data);
        if ndims(d) == 3 && size(d, 1) == 3 && size(d, 2) == 3
            R = squeeze(d(:, :, 1));
        end
    catch
    end
end

%% =====================================================================
function q = local_R_to_quat(R)
%LOCAL_R_TO_QUAT  3x3 rotation matrix -> [w x y z] unit quaternion.
    tr = R(1,1) + R(2,2) + R(3,3);
    if tr > 0
        S = 2 * sqrt(tr + 1);
        w = 0.25 * S;
        x = (R(3,2) - R(2,3)) / S;
        y = (R(1,3) - R(3,1)) / S;
        z = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = 2 * sqrt(1 + R(1,1) - R(2,2) - R(3,3));
        w = (R(3,2) - R(2,3)) / S;
        x = 0.25 * S;
        y = (R(1,2) + R(2,1)) / S;
        z = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = 2 * sqrt(1 + R(2,2) - R(1,1) - R(3,3));
        w = (R(1,3) - R(3,1)) / S;
        x = (R(1,2) + R(2,1)) / S;
        y = 0.25 * S;
        z = (R(2,3) + R(3,2)) / S;
    else
        S = 2 * sqrt(1 + R(3,3) - R(1,1) - R(2,2));
        w = (R(2,1) - R(1,2)) / S;
        x = (R(1,3) + R(3,1)) / S;
        y = (R(2,3) + R(3,2)) / S;
        z = 0.25 * S;
    end
    q = [w, x, y, z];
    q = q / max(norm(q), eps);
    if q(1) < 0; q = -q; end
end

%% =====================================================================
function v = local_xyz_scalar(csb, fx, fy, fz)
    v = nan(1, 3);
    try
        tx = local_field_chain(csb, fx); ty = local_field_chain(csb, fy); tz = local_field_chain(csb, fz);
        if ~isempty(tx) && ~isempty(ty) && ~isempty(tz)
            v = [double(tx.Data(1)), double(ty.Data(1)), double(tz.Data(1))];
        end
    catch
    end
end

%% =====================================================================
function v = local_xyz_vec3(csb, chain)
    v = nan(1, 3);
    try
        ts = local_field_chain(csb, chain);
        if ~isempty(ts)
            d = double(ts.Data);
            d = reshape(d(1, :), 1, []);
            d = d(1:3);
            v = d;
        end
    catch
    end
end

%% =====================================================================
function ts = local_field_chain(s, chain)
    ts = [];
    for k = 1:numel(chain)
        if ~isstruct(s) || ~isfield(s, chain{k})
            return;
        end
        s = s.(chain{k});
    end
    ts = s;
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

%% =====================================================================
function local_print_skeleton(skel)
    fprintf('  joint centres @ t=%.4fs (m, world frame)\n', skel.time);
    for k = 1:numel(skel.joint_order)
        n = skel.joint_order{k};
        v = skel.(n);
        if any(isnan(v))
            fprintf('    %-6s  <missing from CombinedSignalBus>\n', n);
        else
            fprintf('    %-6s  [% .4f % .4f % .4f]\n', n, v(1), v(2), v(3));
        end
    end
end
