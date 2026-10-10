function inv = simscape_model_inventory(mdl)
%SIMSCAPE_MODEL_INVENTORY  Joint and mass inventory of a loaded Simscape model.
%
%   INV = SIMSCAPE_MODEL_INVENTORY(MDL) walks every sm_lib block of the
%   loaded Simscape Multibody model MDL (#11569 task 3) and returns:
%     INV.joints: struct array (path, type, dof) for every sm_lib joint.
%     INV.bodies: struct array (path, kind, shape, mass_kg, moments_kgm2,
%       mass_basis) for every solid and inertia block.  mass_kg is NaN
%       (JSON null) when it cannot be evaluated: unavailable is never zero.
%     INV.n_coordinates: sum of joint DOF.
%   Block parameters are resolved in the block's own scope (slResolve) and
%   converted to SI with simscape.Value, so g / lbm / in entries are exact.
%   moments_kgm2 are the principal moments about the solid's centre for a
%   uniform cylinder, sphere or brick, or the block's own moments for
%   custom inertia; NaN when the geometry is not one of those.
%
%   Preconditions: MDL is loaded (load_system) and contains sm_lib blocks.
%   Postconditions: the model is not modified or saved.

    arguments
        mdl (1,:) char
    end
    assert(bdIsLoaded(mdl), 'NotLoaded: load_system(''%s'') first', mdl);
    raw_blocks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'Type', 'Block');
    refs = strrep(get_param(raw_blocks, 'ReferenceBlock'), newline, ' ');
    keep = startsWith(refs, 'sm_lib/');
    raw_blocks = raw_blocks(keep);
    blocks = strrep(raw_blocks, newline, ' ');
    refs = refs(keep);

    is_joint = startsWith(refs, 'sm_lib/Joints/');
    inv.joints = arrayfun(@(k) local_joint(blocks{k}, refs{k}), find(is_joint));
    is_mass = startsWith(refs, 'sm_lib/Body Elements/') & ...
        ~contains(refs, 'Sensor') & ~contains(refs, 'Graphic');
    idx = find(is_mass);
    inv.bodies = arrayfun(@(k) local_body(raw_blocks{k}, blocks{k}, refs{k}), idx);
    inv.n_coordinates = sum([inv.joints.dof]);
    assert(numel(inv.joints) > 0, 'NoJoints: %s has no sm_lib joints', mdl);
end

function j = local_joint(path, ref)
    dof = containers.Map( ...
        {'Weld Joint', 'Revolute Joint', 'Prismatic Joint', 'Universal Joint', ...
         'Cylindrical Joint', 'Gimbal Joint', 'Spherical Joint', 'Planar Joint', ...
         'Rectangular Joint', 'Cartesian Joint', 'Bearing Joint', ...
         'Telescoping Joint', 'Bushing Joint', '6-DOF Joint', 'Pin Slot Joint', ...
         'Constant Velocity Joint', 'Lead Screw Joint'}, ...
        {0, 1, 1, 2, 2, 3, 3, 3, 2, 3, 4, 4, 6, 6, 2, 2, 1});
    type = extractAfter(ref, 'sm_lib/Joints/');
    assert(isKey(dof, type), 'UnknownJoint: %s (%s)', type, path);
    j = struct('path', path, 'type', type, 'dof', dof(type));
end

function b = local_body(raw, path, ref)
    kind = extractAfter(ref, 'sm_lib/Body Elements/');
    b = struct('path', path, 'kind', kind, 'shape', '', 'mass_kg', NaN, ...
        'moments_kgm2', [NaN NaN NaN], 'mass_basis', '');
    params = fieldnames(get_param(raw, 'DialogParameters'));
    inertia_type = 'Custom';
    if any(strcmp(params, 'InertiaType'))
        inertia_type = get_param(raw, 'InertiaType');
    end
    [b.shape, volume, unit_moments] = local_geometry(raw, kind, params);
    if any(strcmp(inertia_type, {'Custom', 'PointMass'})) || strcmp(kind, 'Inertia')
        b.mass_kg = local_value(raw, 'Mass', 'kg');
        b.mass_basis = 'mass';
        if any(strcmp(params, 'MomentsOfInertia')) && ~strcmp(inertia_type, 'PointMass')
            b.moments_kgm2 = reshape(local_value(raw, 'MomentsOfInertia', 'kg*m^2'), 1, 3);
        elseif strcmp(inertia_type, 'PointMass')
            b.moments_kgm2 = [0 0 0];
        end
        return
    end
    % CalculateFromGeometry: mass from Mass or Density x volume.
    if strcmp(get_param(raw, 'BasedOnType'), 'Mass')
        b.mass_kg = local_value(raw, 'Mass', 'kg');
        b.mass_basis = 'mass';
    else
        b.mass_kg = local_value(raw, 'Density', 'kg/m^3') * volume;
        b.mass_basis = 'density';
    end
    b.moments_kgm2 = b.mass_kg * unit_moments;
end

function [shape, volume, unit_moments] = local_geometry(raw, kind, params)
% Uniform-solid volume and principal moments per unit mass (solid frame).
    shape = 'unsupported';
    volume = NaN;
    unit_moments = [NaN NaN NaN];
    if strcmp(kind, 'Cylindrical Solid')
        r = local_value(raw, 'CylinderRadius', 'm');
        h = local_value(raw, 'CylinderLength', 'm');
        shape = 'cylinder';
        volume = pi * r^2 * h;
        t = (3 * r^2 + h^2) / 12;
        unit_moments = [t t r^2 / 2];
    elseif strcmp(kind, 'Spherical Solid')
        r = local_value(raw, 'SphereRadius', 'm');
        shape = 'sphere';
        volume = 4 / 3 * pi * r^3;
        unit_moments = 2 / 5 * r^2 * [1 1 1];
    elseif strcmp(kind, 'Brick Solid') && any(strcmp(params, 'BrickDimensions'))
        d = reshape(local_value(raw, 'BrickDimensions', 'm'), 1, 3);
        shape = 'brick';
        volume = prod(d);
        unit_moments = [d(2)^2 + d(3)^2, d(1)^2 + d(3)^2, d(1)^2 + d(2)^2] / 12;
    elseif strcmp(kind, 'Inertia')
        shape = 'none';
    end
end

function v = local_value(raw, name, si_unit)
% Resolve NAME in the block scope and convert from its *Units to SI_UNIT.
    v = slResolve(get_param(raw, name), raw);
    unit_param = [name 'Units'];
    if any(strcmp(fieldnames(get_param(raw, 'DialogParameters')), unit_param))
        v = value(simscape.Value(double(v), get_param(raw, unit_param)), si_unit);
    end
    v = double(v);
end
