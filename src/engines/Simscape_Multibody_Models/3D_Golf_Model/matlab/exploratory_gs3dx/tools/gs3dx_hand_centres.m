function [hand, ikf] = gs3dx_hand_centres(ik)
%GS3DX_HAND_CENTRES  Lead and trail hand-sphere centres from whole-body IK (#11160).
%
%   [HAND, IKF] = GS3DX_HAND_CENTRES(IK) returns the Grip/LHand/R and
%   Grip/RHand/R positions in the model World frame, 3 x 2 x numel(IKF),
%   for IK frames whose grip loop closed (IK.status >= 1).  IKF lists the
%   column indices into IK.frames / IK.joint.

    assert(all(isfield(ik, {'frames', 'joint', 'joint_ids', 'model', 'status'})), 'gs3dx:club', ...
        'IK must be a GS3DX_WHOLE_BODY_IK result');
    mdl = ik.model;
    load_system(mdl);
    wf = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'ReferenceBlock', 'sm_lib/Frames and Transforms/World Frame');
    ks = simscape.multibody.KinematicsSolver(mdl);
    ids = string(ks.jointPositionVariables.ID);
    assert(isequal(ids(:), string(ik.joint_ids(:))), 'gs3dx:club', 'IK.joint_ids do not match %s', mdl);
    addFrameVariables(ks, 'lead', 'Translation', [wf{1} '/W'], [mdl '/Grip/LHand/R']);
    addFrameVariables(ks, 'trail', 'Translation', [wf{1} '/W'], [mdl '/Grip/RHand/R']);
    roles = gs3dx_ik_joint_roles(ks.jointPositionVariables);
    closed = roles.closed_mask;
    addTargetVariables(ks, ids(~closed));
    addOutputVariables(ks, string(ks.frameVariables.ID));
    addInitialGuessVariables(ks, ids(closed));
    ikf = find(ik.status >= 1);
    hand = zeros(3, 2, numel(ikf));
    for i = 1:numel(ikf)
        [o, status] = solve(ks, ik.joint(~closed, ikf(i)), ik.joint(closed, ikf(i)));
        assert(status >= 1 && all(isfinite(o(1:6))), 'gs3dx:club:invalid_hand_pose', ...
            'Hand-centre reconstruction failed at capture frame %d', ik.frames(ikf(i)));
        hand(:, :, i) = reshape(o(1:6), 3, 2);
    end
end
