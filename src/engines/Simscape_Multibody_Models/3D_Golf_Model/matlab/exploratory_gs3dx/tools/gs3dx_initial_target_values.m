function targets = gs3dx_initial_target_values(keys, units, q)
%GS3DX_INITIAL_TARGET_VALUES  Pure kinematic mapper from solved IK joint states to native initial targets (#10979).
%
%   TARGETS = GS3DX_INITIAL_TARGET_VALUES(KEYS, UNITS, Q) maps a single 48-variable
%   solved joint pose Q into native Simulink workspace initialization parameters
%   and literal block position targets for the GS3DX_Human model.
%
%   Inputs:
%     KEYS   48-element string or cellstr array of model-relative joint keys
%            formatted as 'BlockPath|Primitive' from GS3DX_JOINT_KEYS.
%     UNITS  48-element string or cellstr array of physical units ('m', 'deg', '1')
%            matching each joint coordinate.
%     Q      48-element finite real numeric vector of solved joint positions.
%
%   Outputs:
%     TARGETS struct with fields:
%       .workspace_values       struct containing 66 native workspace start parameters
%                               (33 positions and 33 explicit zero velocities).
%       .block_position_values  4x1 struct array with fields (block, primitive, value, unit)
%                               for literal neck (Rx/Ry) and midfoot (L/R Rz) joints.
%       .qualification          'PURE_MAPPING_ONLY'
%
%   Design Contracts & Tolerances:
%     - Nonzero spherical joint angles require a unit axis with tolerance |norm(u)-1| <= 1e-4.
%     - Zero spherical angles (|theta| <= 1e-12 deg) permit arbitrary axes including [0 0 0].
%     - Intrinsic follower X-Y-Z Euler extraction explicitly fails with
%       'gs3dx:target_values:gimbal_singularity' if |cos(middle_angle)| <= 1e-6.
%     - Pure mapping only: no Simulink model mutation or simulation is performed.

    arguments
        keys
        units
        q
    end

    % 1. Validate keys argument and enforce whitelist via unified schema
    keys_str = string(keys);
    if numel(keys_str) ~= 48
        error('gs3dx:target_values:invalid_keys', ...
            'KEYS must have exactly 48 elements.');
    end
    keys_str = keys_str(:);

    if numel(unique(keys_str)) ~= 48
        error('gs3dx:target_values:invalid_keys', ...
            'KEYS contains duplicate entries.');
    end

    [canonical_keys, expected_units, schema] = local_schema();

    [found, loc] = ismember(canonical_keys, keys_str);
    if ~all(found) || any(~ismember(keys_str, canonical_keys))
        error('gs3dx:target_values:invalid_keys', ...
            'KEYS does not match the required 22-block / 48-variable GS3DX_Human whitelist.');
    end

    % 2. Validate units argument
    units_str = string(units);
    if numel(units_str) ~= 48
        error('gs3dx:target_values:invalid_units', ...
            'UNITS must have exactly 48 elements.');
    end
    units_str = units_str(:);

    actual_units = units_str(loc);
    mismatched_units = (actual_units ~= expected_units);
    if any(mismatched_units)
        error('gs3dx:target_values:invalid_units', ...
            'Incorrect unit for key: %s (expected %s, got %s)', ...
            canonical_keys(find(mismatched_units, 1)), ...
            expected_units(find(mismatched_units, 1)), ...
            actual_units(find(mismatched_units, 1)));
    end

    % 3. Validate pose vector Q: must be a real, finite, 48-element numeric vector
    if ~isnumeric(q) || ~isreal(q) || ~isvector(q) || any(~isfinite(q(:))) || numel(q) ~= 48
        error('gs3dx:target_values:invalid_q', ...
            'Q must be a real, finite, 48-element numeric vector.');
    end
    q_vec = double(q(:));

    % 4. Align solved values to canonical schema order (schema index lookup)
    val = q_vec(loc);

    % 5. Assemble workspace_values struct and literal block_position_values
    ws = struct();
    bp = struct('block', cell(4, 1), 'primitive', cell(4, 1), 'value', cell(4, 1), 'unit', cell(4, 1));
    bp_idx = 0;

    for i = 1:numel(schema)
        b = schema(i);
        v = val(b.indices);
        switch b.target_type
            case 'pelvis'
                % Translation (m)
                ws.TranslationStartPositionX = v(1);
                ws.TranslationStartPositionY = v(2);
                ws.TranslationStartPositionZ = v(3);
                ws.TranslationStartVelocityX = 0.0;
                ws.TranslationStartVelocityY = 0.0;
                ws.TranslationStartVelocityZ = 0.0;
                % Orientation (deg): intrinsic follower X-Y-Z Euler from spherical axis-angle
                hip_xyz = local_spherical_to_euler(v(4:6), v(7));
                ws.HipStartPositionX = hip_xyz(1);
                ws.HipStartPositionY = hip_xyz(2);
                ws.HipStartPositionZ = hip_xyz(3);
                ws.HipStartVelocityX = 0.0;
                ws.HipStartVelocityY = 0.0;
                ws.HipStartVelocityZ = 0.0;

            case 'workspace_scalars'
                for k = 1:numel(b.pos_fields)
                    ws.(b.pos_fields(k)) = v(k);
                    ws.(b.vel_fields(k)) = 0.0;
                end

            case 'spherical_scalars'
                s_xyz = local_spherical_to_euler(v(1:3), v(4));
                for k = 1:3
                    ws.(b.pos_fields(k)) = s_xyz(k);
                    ws.(b.vel_fields(k)) = 0.0;
                end

            case 'universal_vector'
                ws.(b.pos_fields(1)) = [v(1); v(2)];
                ws.(b.vel_fields(1)) = zeros(2, 1);

            case 'spherical_vector'
                h_xyz = local_spherical_to_euler(v(1:3), v(4));
                ws.(b.pos_fields(1)) = h_xyz;
                ws.(b.vel_fields(1)) = zeros(3, 1);

            case 'literal'
                for k = 1:numel(b.primitives)
                    bp_idx = bp_idx + 1;
                    bp(bp_idx).block = char(b.block);
                    bp(bp_idx).primitive = char(b.primitives(k));
                    bp(bp_idx).value = v(k);
                    bp(bp_idx).unit = 'deg';
                end
        end
    end

    % 6. Assemble final return struct
    targets = struct( ...
        'workspace_values', ws, ...
        'block_position_values', bp, ...
        'qualification', 'PURE_MAPPING_ONLY');
end

function [canonical_keys, expected_units, schema] = local_schema()
% 22-block explicit declarative schema for GS3DX_Human initial target mapping.
% Defines block topologies, primitives, physical units, and target dispatch.
    schema = struct(...
        'block', {}, ...
        'primitives', {}, ...
        'units', {}, ...
        'target_type', {}, ...
        'pos_fields', {}, ...
        'vel_fields', {}, ...
        'indices', {});

    function add_entry(blk, prims, un, ttype, pos, vel)
        entry = struct(...
            'block', blk, ...
            'primitives', prims, ...
            'units', un, ...
            'target_type', ttype, ...
            'pos_fields', pos, ...
            'vel_fields', vel, ...
            'indices', []);
        schema(end + 1) = entry;
    end

    % 1. Pelvis (Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint)
    add_entry("Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint", ...
        ["Px.p", "Py.p", "Pz.p", "S.ax_x", "S.ax_y", "S.ax_z", "S.q"], ...
        ["m", "m", "m", "1", "1", "1", "deg"], ...
        "pelvis", ...
        ["TranslationStartPositionX", "TranslationStartPositionY", "TranslationStartPositionZ", ...
         "HipStartPositionX", "HipStartPositionY", "HipStartPositionZ"], ...
        ["TranslationStartVelocityX", "TranslationStartVelocityY", "TranslationStartVelocityZ", ...
         "HipStartVelocityX", "HipStartVelocityY", "HipStartVelocityZ"]);

    % 2. Neck Joint
    add_entry("Hips and Torso Inputs/Neck Joint", ...
        ["Rx.q", "Ry.q"], ["deg", "deg"], "literal", strings(0, 1), strings(0, 1));

    % 3. Spine Tilt
    add_entry("Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint", ...
        ["Rx.q", "Ry.q"], ["deg", "deg"], "workspace_scalars", ...
        ["SpineStartPositionX", "SpineStartPositionY"], ...
        ["SpineStartVelocityX", "SpineStartVelocityY"]);

    % 4. Torso
    add_entry("Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint/Kinetically Driven Revolute", ...
        "Rz.q", "deg", "workspace_scalars", "TorsoStartPosition", "TorsoStartVelocity");

    % 5-9. Left Upper Extremities
    add_upper_extremity("Left", "L");

    % 10-13. Left Lower Extremities
    add_lower_extremity("Left", "L");

    % 14-17. Right Lower Extremities
    add_lower_extremity("Right", "R");

    % 18-22. Right Upper Extremities
    add_upper_extremity("Right", "R");

    function add_upper_extremity(side, pfx)
        % Elbow (Revolute)
        add_entry(side + " Elbow Joint/Revolute Joint/Kinetically Driven Revolute", ...
            "Rz.q", "deg", "workspace_scalars", pfx + "EStartPosition", pfx + "EStartVelocity");
        % Forearm (Revolute)
        add_entry(side + " Forearm/Revolute Joint/Kinetically Driven Revolute", ...
            "Rz.q", "deg", "workspace_scalars", pfx + "FStartPosition", pfx + "FStartVelocity");
        % Scapula (Universal)
        add_entry(side + " Scapula Joint/Universal Joint/Kinetically Driven Universal Joint", ...
            ["Rx.q", "Ry.q"], ["deg", "deg"], "workspace_scalars", ...
            [pfx + "ScapStartPositionX", pfx + "ScapStartPositionY"], ...
            [pfx + "ScapStartVelocityX", pfx + "ScapStartVelocityY"]);
        % Shoulder (Spherical -> intrinsic follower XYZ Euler)
        add_entry(side + " Shoulder Joint/Gimbal Joint/Kinetically Driven", ...
            ["S.ax_x", "S.ax_y", "S.ax_z", "S.q"], ["1", "1", "1", "deg"], "spherical_scalars", ...
            [pfx + "SStartPositionX", pfx + "SStartPositionY", pfx + "SStartPositionZ"], ...
            [pfx + "SStartVelocityX", pfx + "SStartVelocityY", pfx + "SStartVelocityZ"]);
        % Wrist (Universal)
        add_entry(side + " Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint", ...
            ["Rx.q", "Ry.q"], ["deg", "deg"], "workspace_scalars", ...
            [pfx + "WStartPositionX", pfx + "WStartPositionY"], ...
            [pfx + "WStartVelocityX", pfx + "WStartVelocityY"]);
    end

    function add_lower_extremity(side, pfx)
        % Midfoot (Literal)
        add_entry("Lower Body/" + pfx + " Midfoot Joint", ...
            "Rz.q", "deg", "literal", strings(0, 1), strings(0, 1));
        % Ankle (Universal -> 2x1 vector)
        add_entry("Lower Body/" + side + " Ankle Joint/Kinetically Driven Universal Joint", ...
            ["Rx.q", "Ry.q"], ["deg", "deg"], "universal_vector", ...
            pfx + "AnkleStartPosition", pfx + "AnkleStartVelocity");
        % Hip (Spherical -> 3x1 vector)
        add_entry("Lower Body/" + side + " Hip Joint/Kinetically Driven", ...
            ["S.ax_x", "S.ax_y", "S.ax_z", "S.q"], ["1", "1", "1", "deg"], "spherical_vector", ...
            pfx + "HipStartPosition", pfx + "HipStartVelocity");
        % Knee (Revolute -> scalar)
        add_entry("Lower Body/" + side + " Knee Joint/Kinetically Driven Revolute", ...
            "Rz.q", "deg", "workspace_scalars", ...
            pfx + "KneeStartPosition", pfx + "KneeStartVelocity");
    end

    % Assign linear indices 1:48 and compile canonical_keys & expected_units
    canonical_keys = strings(48, 1);
    expected_units = strings(48, 1);
    cur_idx = 0;
    for i = 1:numel(schema)
        n = numel(schema(i).primitives);
        schema(i).indices = cur_idx + (1:n);
        for k = 1:n
            cur_idx = cur_idx + 1;
            canonical_keys(cur_idx) = schema(i).block + "|" + schema(i).primitives(k);
            expected_units(cur_idx) = schema(i).units(k);
        end
    end
end

function euler_deg = local_spherical_to_euler(axis, angle_deg)
% Convert axis-angle to intrinsic follower X-Y-Z Euler angles [deg].
% Rodrigues formula + canonical Euler extraction.
    tol_axis = 1e-4;
    tol_zero_ang = 1e-12;
    tol_singularity = 1e-6;

    if abs(angle_deg) <= tol_zero_ang
        euler_deg = [0.0; 0.0; 0.0];
        return;
    end

    ax = double(axis(:));
    norm_ax = norm(ax);
    if abs(norm_ax - 1.0) > tol_axis
        error('gs3dx:target_values:invalid_axis', ...
            'Nonzero spherical joint angle requires unit rotation axis (tolerance %g).', tol_axis);
    end
    u = ax / norm_ax;

    ang_rad = deg2rad(angle_deg);
    c = cos(ang_rad);
    s = sin(ang_rad);
    C = 1.0 - c;

    % Rodrigues formula: R = c*I + s*K + (1-c)*(u*u')
    K = [  0.0,  -u(3),   u(2); ...
          u(3),    0.0,  -u(1); ...
         -u(2),   u(1),    0.0];
    R = c * eye(3) + s * K + C * (u * u.');

    % Validate proper rotation
    det_R = det(R);
    ortho_err = norm(R.' * R - eye(3), 'fro');
    if abs(det_R - 1.0) > 1e-4 || ortho_err > 1e-4
        error('gs3dx:target_values:invalid_axis', ...
            'Constructed rotation matrix is not a valid SO(3) element.');
    end

    % Intrinsic follower X-Y-Z Euler extraction: R = Rx(a) * Ry(b) * Rz(c)
    % R(1,3) = sin(b), hypot(R(1,1), R(1,2)) = cos(b) >= 0
    cb = hypot(R(1, 1), R(1, 2));
    sb = R(1, 3);

    if cb <= tol_singularity
        error('gs3dx:target_values:gimbal_singularity', ...
            'Intrinsic X-Y-Z Euler extraction encountered gimbal lock (|cos(b)| <= %g).', tol_singularity);
    end

    b_rad = atan2(sb, cb);
    a_rad = atan2(-R(2, 3), R(3, 3));
    c_rad = atan2(-R(1, 2), R(1, 1));

    euler_deg = rad2deg([a_rad; b_rad; c_rad]);
end
