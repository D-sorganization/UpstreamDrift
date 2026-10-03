function pose = gs3dx_reference_pose(ik, ref, opts)
%GS3DX_REFERENCE_POSE  Render pose with the legs the simulation tracks (#10979).
%
%   POSE = GS3DX_REFERENCE_POSE(IK, REF) returns the IK struct IK
%   (GS3DX_WHOLE_BODY_IK) with its pelvis and leg joints replaced by the
%   leg reference REF (GS3DX_LEG_REFERENCE) at the same capture frames, for
%   GS3DX_RENDER.  The whole-body IK has no toe target, so its ankle angles
%   leave the feet free to spin about the shank (at address both toes
%   pointed away from the ball); REF sets each foot from the capture's toe
%   markers and is what the leg servos track.  The upper body keeps the IK
%   angles.  Rows are named by GS3DX_JOINT_KEYS, so POSE renders on any
%   GS3DX variant.
%
%   REF.q is [hip X Y Z, knee, ankle X Y] per leg, L then R, in degrees
%   (GS3DX_LEG_FK); the hips and the pelvis joint's rotation are Spherical
%   joints, whose position variables are an axis and an angle.  POSE keeps
%   only the IK frames REF covers.
%
%   Option neck: the neck reference (2 x numel(REF.t) rad, a model's
%   NeckReference on LegReferenceTime), which sets the neck joint rows.

    arguments
        ik (1,1) struct
        ref (1,1) struct
        opts.neck double = []
    end
    assert(isfield(ik, 'model') && isfield(ik, 'frames') && isfield(ik, 'joint_ids'), 'gs3dx:refpose', ...
        'IK must carry .model, .frames and .joint_ids (GS3DX_WHOLE_BODY_IK)');
    [ok, at] = ismember(ik.frames, ref.frames);
    assert(any(ok), 'gs3dx:refpose', 'REF covers none of the IK frames');
    n = size(ik.joint, 2);
    pose = ik;
    for g = string(fieldnames(ik)).'
        x = ik.(g);
        if (isnumeric(x) || islogical(x)) && size(x, 2) == n && n > 1
            pose.(g) = x(:, ok);
        end
    end
    at = at(ok);
    if ~isfield(pose, 'joint_keys')
        [k0, id0] = gs3dx_joint_keys(char(ik.model));
        [found, r] = ismember(string(ik.joint_ids), id0);
        assert(all(found), 'gs3dx:refpose', 'IK joint IDs are not those of %s', ik.model);
        pose.joint_keys = k0(r);
    end
    keys = string(pose.joint_keys);

    pelvis = "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|";
    pj = ref.pelvis_joint;
    pose = local_set(pose, keys, pelvis + ["Px.p" "Py.p" "Pz.p"], pj.translation(:, at));
    pose = local_set(pose, keys, pelvis + "S." + ["ax_x" "ax_y" "ax_z" "q"], local_spherical(pj.xyz(:, at)));
    if ~isempty(opts.neck)
        assert(size(opts.neck, 1) == 2 && size(opts.neck, 2) == numel(ref.frames), 'gs3dx:refpose', ...
            'NECK must be 2 x %d (rad)', numel(ref.frames));
        neck = "Hips and Torso Inputs/Neck Joint|" + ["Rx.q" "Ry.q"];
        found = ismember(neck, keys);
        if ~all(found)
            pose.joint_keys = [pose.joint_keys(:); neck(:)];
            pose.joint = [pose.joint; zeros(2, size(pose.joint, 2))];
            keys = string(pose.joint_keys);
        end
        pose = local_set(pose, keys, neck, rad2deg(opts.neck(:, at)));
    end
    side = ["Left" "Right"];
    for s = 1:2
        q = ref.q(6 * (s - 1) + (1:6), at);
        leg = "Lower Body/" + side(s);
        pose = local_set(pose, keys, leg + " Hip Joint/Kinetically Driven|S." + ["ax_x" "ax_y" "ax_z" "q"], local_spherical(q(1:3, :)));
        pose = local_set(pose, keys, leg + " Knee Joint/Kinetically Driven Revolute|Rz.q", q(4, :));
        pose = local_set(pose, keys, leg + " Ankle Joint/Kinetically Driven Universal Joint|" + ["Rx.q" "Ry.q"], q(5:6, :));
    end
end

function pose = local_set(pose, keys, names, values)
    [found, r] = ismember(names, keys);
    assert(all(found), 'gs3dx:refpose', 'IK has no joint %s', strjoin(names(~found), ', '));
    pose.joint(r, :) = values;
end

function v = local_spherical(xyz)
% Intrinsic X-Y-Z angles (deg) -> Spherical joint axis and angle (deg).
    v = zeros(4, size(xyz, 2));
    for k = 1:size(xyz, 2)
        a = xyz(:, k);
        R = local_r(1, a(1)) * local_r(2, a(2)) * local_r(3, a(3));
        aa = rotm2axang(R);
        if aa(4) < 1e-12
            aa(1:3) = [0 0 1];
        end
        v(:, k) = [aa(1:3).'; rad2deg(aa(4))];
    end
end

function R = local_r(axis, a)
    c = cosd(a);
    s = sind(a);
    switch axis
        case 1, R = [1 0 0; 0 c -s; 0 s c];
        case 2, R = [c 0 s; 0 1 0; -s 0 c];
        otherwise, R = [c -s 0; s c 0; 0 0 1];
    end
end
