function report = gs3dx_build_fit_legs(info, ref, opts)
%GS3DX_BUILD_FIT_LEGS  Build GS3DX_FitLegs: GS3DX_Fit with leg servo references from the capture.
%
%   REPORT = GS3DX_BUILD_FIT_LEGS(INFO, REF) copies GS3DX_Fit to
%   GS3DX_FitLegs and makes its leg servo track REF (GS3DX_LEG_REFERENCE on
%   GS3DX_Fit, #10979) from capture frame START_FRAME on:
%
%   * Servo.  The 'Lower Body/Leg Torque Commands' Constant (LegTorqueCommand
%     + LegServoKp .* LegAngleReference) becomes a From Workspace block of the
%     same name that plays
%       LegTorqueCommand + LegServoKp .* LegReferenceAngle + LegServoKd .* LegReferenceRate
%     on LegReferenceTime (linear interpolation, final value held), so the
%     servo torque is LegTorqueCommand + Kp (q_ref - q) + Kd (qd_ref - qd).
%     One block for one: the nonvirtual block count is asserted unchanged.
%   * Start state.  Legs (L/R Hip|Knee|Ankle StartPosition/StartVelocity)
%     and pelvis (TranslationStartPosition/Velocity*, HipStartPosition/
%     Velocity*) start on the reference at START_FRAME.  The regression
%     drive file also sets the pelvis variables, so REPORT.start holds every
%     start variable for the caller to pass after the drive (the variables
%     option of GS3DX_CONTACT_CHECK); the same values are saved in the model,
%     each variable and the whole struct as LegReferenceStart.
%     It includes PlaneTilt, the tilt of the pelvis joint's base frame about
%     World X: the drive file sets 22.5 deg, the saved model 30 deg, and the
%     reference (like the whole-body IK) was computed on the saved model.
%   * Ground.  The capture frame is the model World (GS3DX_WHOLE_BODY_IK),
%     so GroundRotation is the identity ([facing, toward target, up]) and
%     GroundOffset is under the two start feet, AnkleHeight below the left
%     ankle as GS3DX_BUILD_CONTACT places it.
%
%   The upper-body drive is unchanged: it is not yet synchronized with the
%   capture (the next step of docs/ANTHROPOMETRY.md).
%
%   Options: overwrite (false), start_frame (REF.frames(1)).
%   REPORT fields: .start (struct of start variables), .ground_p, .frames
%   (capture frames played), .budget (GS3DX_BLOCK_BUDGET, compiled=true).

    arguments
        info (1,1) struct
        ref (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.start_frame (1,1) double {mustBeInteger, mustBePositive} = ref.frames(1)
    end
    names = gs3dx_names();
    src = char(names.variants.fit);
    mdl = char(names.variants.fit_legs);
    assert(strcmp(ref.model, src), 'gs3dx:fitlegs', 'REF was computed on %s, not %s', ref.model, src);
    s = find(ref.frames == opts.start_frame);
    assert(isscalar(s) && s < numel(ref.frames), 'gs3dx:fitlegs', ...
        'Start frame %d is not inside the reference', opts.start_frame);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:fitlegs');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    before = local_nonvirtual(mdl);
    ws = get_param(mdl, 'ModelWorkspace');
    k = s:numel(ref.frames);
    assignin(ws, 'LegReferenceTime', ref.t(k) - ref.t(s));
    assignin(ws, 'LegReferenceAngle', ref.q(:, k));
    assignin(ws, 'LegReferenceRate', ref.qd(:, k));
    report.start = local_start(ref, s);
    report.start.PlaneTilt = local_plane_tilt(ws, ref.pelvis_joint.base.R);
    for f = fieldnames(report.start).'
        assignin(ws, f{1}, report.start.(f{1}));
    end
    assignin(ws, 'LegReferenceStart', report.start);
    report.ground_p = local_ground(ref, s, ws.getVariable('AnkleHeight'));
    assignin(ws, 'GroundRotation', eye(3));
    assignin(ws, 'GroundOffset', report.ground_p);
    local_feedforward([mdl '/Lower Body']);

    after = local_nonvirtual(mdl);
    assert(after == before, 'gs3dx:fitlegs', ...
        'Postcondition: the servo edit changed the block count (%d -> %d)', before, after);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % the sensors GS3DX_CONTACT_CHECK adds in memory
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:fitlegs', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
    report.frames = ref.frames(k);
end

function v = local_start(ref, s)
% Start variables: leg joints (deg, deg/s) and the pelvis joint (m, deg).
    joints = {'Hip', 'Knee', 'Ankle'};
    cols = {1:3, 4, 5:6};
    sides = 'LR';
    for k = 1:2
        for j = 1:3
            rows = (k - 1) * 6 + cols{j};
            v.([sides(k) joints{j} 'StartPosition']) = ref.q(rows, s).';
            v.([sides(k) joints{j} 'StartVelocity']) = ref.qd(rows, s).';
        end
    end
    v.LegAngleReference = ref.q(:, s);
    pj = ref.pelvis_joint;
    xyz = 'XYZ';
    for a = 1:3
        v.(['TranslationStartPosition' xyz(a)]) = pj.translation(a, s);
        v.(['TranslationStartVelocity' xyz(a)]) = pj.translation_rate(a, s);
        v.(['HipStartPosition' xyz(a)]) = pj.xyz(a, s);
        v.(['HipStartVelocity' xyz(a)]) = pj.xyz_rate(a, s);
    end
end

function tilt = local_plane_tilt(ws, base_R)
% The saved PlaneTilt, checked against the pelvis joint base of the reference.
    tilt = ws.getVariable('PlaneTilt');
    if isa(tilt, 'Simulink.Parameter')
        tilt = tilt.Value;
    end
    Rx = [1 0 0; 0 cosd(tilt) -sind(tilt); 0 sind(tilt) cosd(tilt)];
    assert(norm(base_R - Rx) < 1e-6, 'gs3dx:fitlegs', ...
        'The reference pelvis joint base is not Rx(PlaneTilt = %g deg)', tilt);
end

function g = local_ground(ref, s, ankle_height)
% Ground point under the start feet, on the left sole plane (World z up).
    sole = @(P) ref.feet.(P).p(:, s) - ankle_height * ref.feet.(P).R(:, 3, s);
    L = sole('L');
    R = sole('R');
    assert(abs(L(3) - R(3)) < 0.02, 'gs3dx:fitlegs', ...
        'Postcondition: the start soles are %.1f cm apart in height', 100 * abs(L(3) - R(3)));
    g = [(L(1:2) + R(1:2)) / 2; L(3)];
end

function local_feedforward(sys)
% Replace the servo Constant with a From Workspace block of the same name.
    constant = [sys '/Leg Torque Commands'];
    expected = 'LegTorqueCommand + LegServoKp .* LegAngleReference';
    assert(strcmp(get_param(constant, 'BlockType'), 'Constant') && ...
        strcmp(get_param(constant, 'Value'), expected), 'gs3dx:fitlegs', ...
        '%s is not the GS3DX_Fit servo Constant', constant);
    lines = get_param(constant, 'LineHandles');
    dst = get_param(lines.Outport, 'DstPortHandle');
    pos = get_param(constant, 'Position');
    delete_line(lines.Outport);
    delete_block(constant);
    blk = add_block('simulink/Sources/From Workspace', constant, 'Position', pos, ...
        'VariableName', ['[LegReferenceTime(:), (LegTorqueCommand(:) + LegServoKp(:) .* LegReferenceAngle' ...
            ' + LegServoKd(:) .* LegReferenceRate).'']'], ...
        'SampleTime', '0', 'Interpolate', 'on', 'OutputAfterFinalValue', 'Holding final value');
    ph = get_param(blk, 'PortHandles');
    for d = dst(:).'
        add_line(sys, ph.Outport, d, 'autorouting', 'on');
    end
end

function n = local_nonvirtual(mdl)
    n = numel(find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
end
