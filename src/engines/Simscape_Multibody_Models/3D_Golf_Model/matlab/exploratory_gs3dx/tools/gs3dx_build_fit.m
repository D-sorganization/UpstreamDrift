function report = gs3dx_build_fit(info, opts)
%GS3DX_BUILD_FIT  Build GS3DX_Fit: GS3DX_Golfer with segment lengths and grip from the capture.
%
%   REPORT = GS3DX_BUILD_FIT(INFO) copies GS3DX_Golfer to GS3DX_Fit and sets
%   its segment lengths to the capture's joint-centre lengths (#10979):
%   GS3DX_FIT_LENGTHS(GS3DX_CAPTURE_JOINT_CENTRES(GS3DX_CAPTURE_MARKERS())).
%
%   * The regression drive file sets LowerTorsoLength and UpperTorsoLength
%     (to the original 12 in), so the upper-body solids are re-pointed at
%     new Fit* variables (SOLIDS below) that no drive file sets.
%   * ThighLength and ShankLength are already model-workspace variables
%     that only GS3DX_LEG_TABLE sets; they are overwritten here.
%   * The hand-on-grip geometry (GS3DX_FIT_GRIP) re-points the literal grip
%     lengths (GRIP_SOLIDS below) at Fit* variables, and the lead standoff's
%     two frames swap ends, so the lead wrist sits on the other side of the
%     grip from the trail wrist (the palms face each other across it).
%   * The frames follow the solids' geometry, so the joints move with the
%     lengths.  Masses stay the GS3DX_Golfer table; inertia follows the
%     geometry.  Joints, drive, stance, contacts and servo are unchanged.
%
%   The edit is parameter-only: the nonvirtual block count is asserted
%   unchanged.  Options: overwrite (false); jc (a GS3DX_CAPTURE_JOINT_CENTRES
%   struct; default: computed from the default capture); grip (GRIP.vars of
%   GS3DX_FIT_GRIP; default: the 2026-09-27 estimate below, which needs a
%   whole-body IK run to recompute).
%
%   REPORT fields: .vars, .source (GS3DX_FIT_LENGTHS), .grip, .solids,
%   .budget.

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.jc struct = struct([])
        opts.grip struct = local_grip_estimate()
    end
    names = gs3dx_names();
    src = char(names.variants.golfer);
    mdl = char(names.variants.fit);
    jc = opts.jc;
    if isempty(jc)
        jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
    end
    fit = gs3dx_fit_lengths(jc);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:fit');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    before = local_nonvirtual(mdl);
    ws = get_param(mdl, 'ModelWorkspace');
    for f = fieldnames(fit.vars).'
        assignin(ws, f{1}, fit.vars.(f{1}));
    end
    solids = local_solids();
    for k = 1:size(solids, 1)
        blk = [mdl '/' solids{k, 1}];
        assert(getSimulinkBlockHandle(blk) > 0, 'gs3dx:fit', 'Solid not found: %s', blk);
        assert(strcmp(get_param(blk, 'CylinderLength'), solids{k, 2}), 'gs3dx:fit', ...
            '%s length is %s, expected %s', blk, get_param(blk, 'CylinderLength'), solids{k, 2});
        assert(strcmp(get_param(blk, 'CylinderLengthUnits'), 'in'), 'gs3dx:fit', '%s is not in inches', blk);
        set_param(blk, 'CylinderLength', solids{k, 3});
    end
    for f = fieldnames(opts.grip).'
        assignin(ws, f{1}, opts.grip.(f{1}));
    end
    solids = [solids; local_grip_solids()];
    for k = size(solids, 1) - size(local_grip_solids(), 1) + 1:size(solids, 1)
        blk = [mdl '/' solids{k, 1}];
        assert(strcmp(get_param(blk, 'CylinderLength'), solids{k, 2}), 'gs3dx:fit', ...
            '%s length is %s, expected %s', blk, get_param(blk, 'CylinderLength'), solids{k, 2});
        assert(strcmp(get_param(blk, 'CylinderLengthUnits'), 'in'), 'gs3dx:fit', '%s is not in inches', blk);
        set_param(blk, 'CylinderLength', solids{k, 3});
    end
    local_flip_lead_standoff([mdl '/Grip/LHandStandoff']);

    after = local_nonvirtual(mdl);
    assert(after == before, 'gs3dx:fit', ...
        'Postcondition: the length edit changed the block count (%d -> %d)', before, after);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
    report.vars = fit.vars;
    report.source = fit.source;
    report.grip = opts.grip;
    report.solids = solids;
end

function s = local_solids()
%LOCAL_SOLIDS  Upper-body solids: old CylinderLength expression, new one (in).
    T = 'Hips and Torso Inputs/';
    s = {
        [T 'LowerTorso'],              'LowerTorsoLength',     'FitLowerTorsoLength'
        [T 'UpperTorsoBase'],          'UpperTorsoLength*0.2', 'FitUpperTorsoLength*0.2'
        [T 'UpperTorsoTop'],           'UpperTorsoLength*0.8', 'FitUpperTorsoLength*0.8'
        'HubtoLS',                     'HubtoSLength',         'FitHubtoSLength'
        'HubtoRS',                     'HubtoSLength',         'FitHubtoSLength'
        'LUpperArm',                   'UpperArmLength',       'FitUpperArmLength'
        'RUpperArm',                   'UpperArmLength',       'FitUpperArmLength'
        'Left Forearm/LUpperForearm',  '0.5*LowerArmLength',   '0.5*FitLowerArmLength'
        'Left Forearm/LLowerForearm',  '0.5*LowerArmLength',   '0.5*FitLowerArmLength'
        'Right Forearm/RUpperForearm', '0.5*LowerArmLength',   '0.5*FitLowerArmLength'
        'Right Forearm/RLowerForearm', '0.5*LowerArmLength',   '0.5*FitLowerArmLength'
        };
end

function s = local_grip_solids()
%LOCAL_GRIP_SOLIDS  Grip solids (in): old literal length, new expression.  The
%   butt-to-shaft length stays 10.5 in (FitGripToShaft is the remainder).
    s = {
        'Grip/2.5" Butt',        '2.5',                      'FitButtToLeadHand'
        'Grip/1.5" Top',         '1.5',                      '0.5*FitHandSpacing'
        'Grip/1.5" Bottom',      '1.5',                      '0.5*FitHandSpacing'
        'Grip/GhostGripSegment', '5',                        'FitGripToShaft'
        'Grip/LHandStandoff',    'LeftWristStandoffLength',  'FitLeftWristStandoff'
        'Grip/RHandStandoff',    'RightWristStandoffLength', 'FitRightWristStandoff'
        };
end

function g = local_grip_estimate()
%LOCAL_GRIP_ESTIMATE  GS3DX_FIT_GRIP on the default capture (in), 2026-09-27:
%   out-of-sample IK of the downswing (frames f0-90:6:f0) on a GS3DX_Fit
%   copy with the first-pass grip, docs/FIT.md.
    g = struct('FitButtToLeadHand', 3.0809, 'FitHandSpacing', 3.0303, ...
        'FitLeftWristStandoff', 1.4635, 'FitRightWristStandoff', 1.4635);
    g.FitGripToShaft = 10.5 - g.FitButtToLeadHand - g.FitHandSpacing;
end

function local_flip_lead_standoff(blk)
% Swap the ends of the standoff's two frames: the wrist frame moves from the
% top curve to the bottom curve and the grip frame the other way, which
% reverses the standoff direction without changing either frame's axes.
    sf = get_param(blk, 'SerializedFrames');
    pat = '<Name>Left Wrist</Name><Origin><Source>GeometricFeature</Source><FeatureName>top curve</FeatureName>';
    assert(contains(sf, pat) && count(sf, 'bottom curve') == 1, 'gs3dx:fit', ...
        '%s frames are not the GS3DX_Golfer layout', blk);
    sf = strrep(strrep(strrep(sf, 'top curve', '@TOP@'), 'bottom curve', 'top curve'), '@TOP@', 'bottom curve');
    set_param(blk, 'SerializedFrames', sf);
end

function n = local_nonvirtual(mdl)
    n = numel(find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
end
