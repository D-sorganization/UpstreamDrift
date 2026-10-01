function q = gs3dx_simlog_joints(simlog, mdl, jp, t)
%GS3DX_SIMLOG_JOINTS  Simulated joint positions from a Simscape log, as a pose for GS3DX_RENDER (#10979).
%
%   Q = GS3DX_SIMLOG_JOINTS(SIMLOG, MDL, JP, T) reads every joint position
%   variable of MDL's KinematicsSolver (JP, its jointPositionVariables
%   table) from the Simscape log SIMLOG of a run of MDL, resampled onto the
%   times T (s).  Q is an IK-style struct that GS3DX_RENDER accepts:
%     .joint_ids   IDs of JP (such as "j15.Rx.q")
%     .joint_keys  model-independent keys (GS3DX_JOINT_KEYS)
%     .joint       variables x numel(T), in JP's units
%     .t           T
%     .model       MDL
%   Revolute (R*.q) and prismatic (P*.p) variables are read directly.  A
%   spherical primitive logs its quaternion S.Q ([w x y z]); the solver's
%   S.ax_x, S.ax_y, S.ax_z and S.q (deg) are its axis and angle (axis
%   [0 0 1] at zero angle).  Log nodes are found by block name with
%   non-alphanumerics ignored, as Simscape names them.

    arguments
        simlog (1,1)
        mdl (1,:) char
        jp table
        t (1,:) double {mustBeFinite}
    end
    [keys, ids] = gs3dx_joint_keys(mdl, jp);
    unit = string(jp.Unit);
    path = extractAfter(string(jp.BlockPath), strlength(mdl) + 1);
    q = struct('joint_ids', ids, 'joint_keys', keys, 'joint', zeros(numel(ids), numel(t)), 't', t, 'model', mdl);
    cache = containers.Map();
    for k = 1:numel(ids)
        prim = split(extractAfter(ids(k), "."), ".");   % e.g. ["Rz" "q"] or ["S" "ax_x"]
        node = local_node(simlog, path(k));
        if prim(1) == "S"
            if ~isKey(cache, path(k))
                cache(path(k)) = local_axis_angle(node.S.Q.series, t);
            end
            aa = cache(path(k));
            row = find(prim(2) == ["ax_x" "ax_y" "ax_z" "q"]);
            assert(~isempty(row), 'gs3dx:simlog_joints', 'Unknown spherical variable %s', ids(k));
            q.joint(k, :) = aa(row, :);
        else
            s = node.(char(prim(1))).(char(prim(2))).series;
            q.joint(k, :) = interp1(s.time, s.values(char(unit(k))), t, 'linear', 'extrap');
        end
    end
end

function node = local_node(simlog, path)
    node = simlog;
    norm = @(s) lower(regexprep(string(s), '[^A-Za-z0-9]', ''));
    for p = split(path, "/").'
        ids = string(node.childIds);
        hit = ids(norm(ids) == norm(p));
        assert(isscalar(hit), 'gs3dx:simlog_joints', 'No single log node for "%s" in %s', p, path);
        node = node.child(char(hit));
    end
end

function aa = local_axis_angle(series, t)
% [ax_x; ax_y; ax_z; angle (deg)] from logged quaternions [w x y z].
    Q = interp1(series.time, series.values, t, 'linear', 'extrap').';
    Q = Q ./ vecnorm(Q);
    Q = Q .* sign(Q(1, :) + (Q(1, :) == 0));   % w >= 0: angle in [0, 180]
    s = vecnorm(Q(2:4, :));
    ang = 2 * atan2d(s, Q(1, :));
    ax = Q(2:4, :) ./ max(s, eps);
    small = s < 1e-12;
    ax(:, small) = repmat([0; 0; 1], 1, nnz(small));
    aa = [ax; ang];
end
