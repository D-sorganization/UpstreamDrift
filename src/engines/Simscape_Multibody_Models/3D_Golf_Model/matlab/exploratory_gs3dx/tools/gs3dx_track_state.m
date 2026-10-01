function [q, qd] = gs3dx_track_state(log, j, tl, T)
%GS3DX_TRACK_STATE  An upper-body chart's joint angles and rates from a Simscape log (#10979, #11173).
%
%   [Q, QD] = GS3DX_TRACK_STATE(LOG, J, TL, T) reads the joint of one
%   '<prefix> Input Function' chart from the Simscape log LOG and returns
%   its angles (deg) and rates (deg/s), axes x numel(TL), on the times TL:
%   the revolute and universal joint angles, and the shoulders' intrinsic
%   X-Y-Z angles and rates from their quaternion and angular velocity
%   (GS3DX_XYZ_MAP, on the branch of the reference).  J has .block (the
%   joint block path), .axes ('', 'XY' or 'XYZ', GS3DX_UPPER_BODY_JOINTS)
%   and .A (the reference angles, axes x numel(T), deg, on the times T).
%   A revolute or universal angle is moved by whole turns onto the
%   reference's branch at the first sample (a start target may sit 360 deg
%   off).  Used by GS3DX_TRACK_LEARN and GS3DX_FEEDBACK_REPORT.

    node = simscape.logging.findNode(log, j.block);
    assert(~isempty(node), 'gs3dx:trackstate', 'No Simscape log of %s', j.block);
    if numel(j.axes) == 3
        t = node.S.Q.series.time;
        Q = node.S.Q.series.values;
        w = node.S.w.series.values('rad/s');
        ref = interp1(T, j.A.', t, 'linear', 'extrap').';
        ang = zeros(3, numel(t)); rate = zeros(3, numel(t));
        z = zeros(3, 1);
        for i = 1:numel(t)
            [~, ang(:, i), rate(:, i)] = gs3dx_xyz_map(Q(i, :), w(i, :), z, z, z, ref(:, i));
        end
    else
        % the joint's revolute primitives in X, Y, Z order (a hinge may
        % turn about any of them)
        kids = childIds(node);
        prim = {'Rx', 'Ry', 'Rz'};
        prim = prim(ismember(prim, kids));
        assert(numel(prim) == max(1, numel(j.axes)), 'gs3dx:trackstate', ...
            '%s logs primitives {%s}, expected %d revolute', j.block, strjoin(kids, ', '), max(1, numel(j.axes)));
        t = node.(prim{1}).q.series.time;
        ang = zeros(numel(prim), numel(t)); rate = ang;
        for p = 1:numel(prim)
            ang(p, :) = node.(prim{p}).q.series.values('deg');
            rate(p, :) = node.(prim{p}).w.series.values('deg/s');
        end
        ang = ang - 360 * round((ang(:, 1) - j.A(:, 1)) / 360);
    end
    [t, iu] = unique(t);
    q = interp1(t, ang(:, iu).', tl(:), 'linear', 'extrap').';
    qd = interp1(t, rate(:, iu).', tl(:), 'linear', 'extrap').';
end
