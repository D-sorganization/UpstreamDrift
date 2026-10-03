function ref = gs3dx_upper_body_reference(ik, opts)
%GS3DX_UPPER_BODY_REFERENCE  Upper-body joint references from a whole-body IK (#10979).
%
%   REF = GS3DX_UPPER_BODY_REFERENCE(IK) turns the KinematicsSolver joint
%   values of a GS3DX_WHOLE_BODY_IK (on GS3DX_Fit) into the angles that the
%   upper-body '<J> Input Function' charts read, for the twelve joints
%   Spine, Torso, L/R Scap, L/R S(houlder), L/R E(lbow), L/R F(orearm) and
%   L/R W(rist):
%
%   * revolute (Torso, E, F) and universal (Spine, Scap, W) joints: the
%     primitive angles (Rz; Rx, Ry) in degrees, unwrapped to one continuous
%     branch;
%   * spherical (shoulders): the intrinsic X-Y-Z angles of the joint
%     rotation (GS3DX_XYZ_MAP, the convention of the charts' 'XYZ
%     Kinematics'), each frame on the 360-degree branch of the previous one.
%
%   Every angle is filtered (zero-phase Butterworth, order 4, CUTOFF_HZ) and
%   differentiated (gradient) for the rates.
%
%   Options: cutoff_hz (12).
%   REF fields:
%     .model, .frames, .t (s, from the first frame), .rate_hz, .cutoff_hz
%     .joints     struct array: .prefix ('LS', ...), .axes ('XYZ', 'XY' or
%                 ''), .ids (solver variables), .angle and .rate
%                 (numel(axes) x frames, deg and deg/s; one row for '')
%     .start      struct of <prefix>StartPosition/Velocity[<axis>] at the
%                 first frame, the model-workspace start variables

    arguments
        ik (1,1) struct
        opts.cutoff_hz (1,1) double {mustBePositive} = 12
    end
    frames = ik.frames;
    n = numel(frames);
    assert(n >= 16 && all(diff(frames) == frames(2) - frames(1)), 'gs3dx:ubref', ...
        'Precondition: IK frames must be at least 16 and evenly spaced');
    assert(all(ik.status >= 1), 'gs3dx:ubref', 'Precondition: the IK did not close the loop in every frame');
    t = ik.t(:).' - ik.t(1);
    rate = (n - 1) / t(end);
    assert(opts.cutoff_hz < rate / 2, 'gs3dx:ubref', 'Cutoff above Nyquist (%g Hz)', rate / 2);
    [b, a] = butter(4, opts.cutoff_hz / (rate / 2));

    ref = struct('model', ik.model, 'frames', frames, 't', t, 'rate_hz', rate, ...
        'cutoff_hz', opts.cutoff_hz);
    spec = gs3dx_upper_body_joints();
    for k = 1:numel(spec)
        j.prefix = spec(k).prefix;
        j.axes = spec(k).axes;
        j.ids = local_ids(ik, spec(k).id, j.axes);
        raw = local_angles(ik, j);
        j.angle = filtfilt(b, a, raw.').';
        j.rate = gradient(j.angle, 1 / rate);
        joints(k) = j; %#ok<AGROW> twelve joints
    end
    ref.joints = joints;
    ref.start = local_start(joints);
end

function ids = local_ids(ik, j, axes)
    switch numel(axes)
        case 0
            ids = {[j '.Rz.q']};
        case 2
            ids = {[j '.Rx.q'], [j '.Ry.q']};
        otherwise
            ids = strcat(j, {'.S.ax_x', '.S.ax_y', '.S.ax_z', '.S.q'});
    end
    missing = setdiff(ids, ik.joint_ids);
    assert(isempty(missing), 'gs3dx:ubref', 'The IK has no joint variable %s', strjoin(missing, ', '));
end

function ang = local_angles(ik, j)
    [~, rows] = ismember(j.ids, ik.joint_ids);
    v = ik.joint(rows, :);
    if numel(j.axes) < 3   % one continuous branch: the IK may wrap an angle to (-180, 180]
        ang = rad2deg(unwrap(deg2rad(v), [], 2));
        return
    end
    ang = zeros(3, size(v, 2));
    prev = zeros(3, 1);
    z = zeros(3, 1);
    for f = 1:size(v, 2)
        axis = v(1:3, f) / norm(v(1:3, f));
        Q = [cosd(v(4, f) / 2); sind(v(4, f) / 2) * axis];
        [~, prev] = gs3dx_xyz_map(Q, z, z, z, z, prev);
        ang(:, f) = prev;
    end
end

function s = local_start(joints)
    s = struct();
    for j = joints
        if isempty(j.axes)
            s.([j.prefix 'StartPosition']) = j.angle(1, 1);
            s.([j.prefix 'StartVelocity']) = j.rate(1, 1);
            continue
        end
        for a = 1:numel(j.axes)
            s.([j.prefix 'StartPosition' j.axes(a)]) = j.angle(a, 1);
            s.([j.prefix 'StartVelocity' j.axes(a)]) = j.rate(a, 1);
        end
    end
end
