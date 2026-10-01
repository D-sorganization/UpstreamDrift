function report = gs3dx_build_neck(info, opts)
%GS3DX_BUILD_NECK  Build GS3DX_Neck: GS3DX_Shape with a motion-driven neck (#10979).
%
%   REPORT = GS3DX_BUILD_NECK(INFO) copies GS3DX_Shape to GS3DX_Neck and
%   gives the head a neck.  In GS3DX_Shape the neck and head are rigid with
%   the upper trunk, so the head follows the trunk's turn and rises about
%   50 mm in the downswing where the golfer's drops (docs/SHAPE.md).
%
%   * Joint.  A Universal Joint between Rigid Transform5 and the Neck's
%     "Bottom of Neck" frame, which is the top end of UpperTorsoTop:
%     Rigid Transform5 joins the two with no offset, so the joint pivots at
%     the base of the neck and at zero angles every frame is where it was.
%     The neck frame has the upper trunk's orientation, z along the spine;
%     the joint turns the neck and head about x, then y, the two axes
%     across the spine, which carry the head's centre.  The turn about z
%     is left out: the head is an ellipsoid of revolution about it with
%     its centre on it, so that turn neither shows nor moves mass.
%   * Motion.  Both angles are driven: NeckReference (2 x numel(
%     LegReferenceTime), rad, model workspace) through a From Workspace
%     block on LegReferenceTime and two Simulink-PS Converters with
%     second-order input filtering (time constant NeckFilterTime), which
%     provide the derivatives the joint needs.  The joint computes its
%     torque.
%   * Blocks.  The neck costs 6 compiled blocks.  Four of them are paid for
%     by removing the massless elbow and shoulder spheres (VISUALS, 1
%     compiled block each, 2026-09-28), each checked massless and
%     frame-free before it goes: 973 -> 975.
%
%   Options: overwrite (false), reference (2 x n rad, default zeros: the
%   neck held straight, kinematics those of GS3DX_Shape), filter_time
%   (0.005 s), visuals (the four spheres, paths below the model).
%   REPORT fields: .budget, .joint (block path).

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.reference double = []
        opts.filter_time (1,1) double {mustBePositive} = 0.005
        opts.visuals (1,:) string = ["Left Elbow Joint/Spherical Solid" "Left Shoulder Joint/Spherical Solid1" ...
            "Right Elbow Joint/Spherical Solid1" "Right Shoulder Joint/Spherical Solid1"]
    end
    names = gs3dx_names();
    src = char(names.variants.shape);
    mdl = char(names.variants.neck);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:neck');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    n = numel(ws.getVariable('LegReferenceTime'));
    ref = opts.reference;
    if isempty(ref)
        ref = zeros(2, n);
    end
    assert(isequal(size(ref), [2 n]) && all(isfinite(ref), 'all'), 'gs3dx:neck', ...
        'REFERENCE must be finite, 2 x %d (rad)', n);
    assignin(ws, 'NeckReference', ref);
    assignin(ws, 'NeckFilterTime', opts.filter_time);

    local_remove_visuals(mdl, opts.visuals);
    report.joint = local_insert_neck_joint([mdl '/Hips and Torso Inputs']);

    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % as GS3DX_BUILD_FIT_BALANCE: room for GS3DX_CONTACT_CHECK
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:neck', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
end

function joint = local_insert_neck_joint(sys)
% UpperTorsoTop -> Rigid Transform5 -> Universal Joint -> Neck.
    rt = [sys '/Rigid' newline 'Transform5'];
    neck = [sys '/Neck'];
    assert(strcmp(get_param(rt, 'TranslationMethod'), 'None'), 'gs3dx:neck', ...
        '%s is not the zero-offset transform of GS3DX_Shape', rt);
    rtp = get_param(rt, 'PortHandles');
    nkp = get_param(neck, 'PortHandles');
    line = get_param(rtp.RConn(1), 'Line');
    assert(line ~= -1, 'gs3dx:neck', '%s follower is not connected', rt);
    ends = setdiff([get_param(line, 'SrcPortHandle') get_param(line, 'DstPortHandle')'], rtp.RConn(1));
    assert(isequal(ends, nkp.LConn(1)), 'gs3dx:neck', ...
        '%s follower must join only the Neck''s Bottom of Neck frame', rt);
    delete_line(line);

    pos = get_param(neck, 'Position');
    joint = [sys '/Neck Joint'];
    add_block('sm_lib/Joints/Universal Joint', joint, 'Position', pos - [200 0 200 0]);
    f = fieldnames(get_param(joint, 'DialogParameters'));
    motion = f(endsWith(f, 'MotionActuationMode'));
    torque = f(endsWith(f, 'TorqueActuationMode'));
    assert(numel(motion) == 2 && numel(torque) == 2, 'gs3dx:neck', 'Universal Joint: expected two primitives');
    for k = 1:2
        set_param(joint, motion{k}, 'InputMotion', torque{k}, 'ComputedTorque');
    end
    jp = get_param(joint, 'PortHandles');
    assert(numel(jp.LConn) == 3 && numel(jp.RConn) == 1, 'gs3dx:neck', 'Universal Joint ports changed');
    add_line(sys, rtp.RConn(1), jp.LConn(1), 'autorouting', 'on');
    add_line(sys, jp.RConn(1), nkp.LConn(1), 'autorouting', 'on');

    x = pos(1) - 700;
    y = pos(2) + 120;
    from = add_block('simulink/Sources/From Workspace', [sys '/Neck Reference'], ...
        'Position', [x y x + 90 y + 30], ...
        'VariableName', '[LegReferenceTime(:), NeckReference.'']', ...
        'SampleTime', '0', 'Interpolate', 'on', 'OutputAfterFinalValue', 'Holding final value');
    demux = add_block('simulink/Signal Routing/Demux', [sys '/Neck Reference Demux'], 'Outputs', '2', ...
        'Position', [x + 130 y - 10 x + 135 y + 40]);
    add_line(sys, [get_param(from, 'Name') '/1'], [get_param(demux, 'Name') '/1']);
    for k = 1:2
        cv = add_block('nesl_utility/Simulink-PS Converter', sprintf('%s/Neck Angle %d', sys, k), ...
            'Position', [x + 180 y - 20 + 45 * k x + 210 y + 45 * k], ...
            'Unit', 'rad', 'FilteringAndDerivatives', 'filter', 'SimscapeFilterOrder', '2', ...
            'InputFilterTimeConstant', 'NeckFilterTime');
        add_line(sys, sprintf('%s/%d', get_param(demux, 'Name'), k), [get_param(cv, 'Name') '/1']);
        cp = get_param(cv, 'PortHandles');
        add_line(sys, cp.RConn(1), jp.LConn(k + 1), 'autorouting', 'on');
    end
end

function local_remove_visuals(mdl, visuals)
% Delete massless solids with one frame port (graphics only).
    audit = gs3dx_inertia_audit(mdl);
    for v = visuals
        blk = [mdl '/' char(v)];
        row = string(audit.solids.block) == v;
        assert(nnz(row) == 1 && audit.solids.mass(row) < 1e-12, 'gs3dx:neck', ...
            '%s is not a massless solid', blk);
        ph = get_param(blk, 'PortHandles');
        assert(isscalar([ph.LConn ph.RConn]), 'gs3dx:neck', '%s carries more than its own frame', blk);
        delete_block(blk);
    end
end
