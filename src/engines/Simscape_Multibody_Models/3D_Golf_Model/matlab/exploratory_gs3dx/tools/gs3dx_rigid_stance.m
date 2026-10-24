function report = gs3dx_rigid_stance(info, opts)
%GS3DX_RIGID_STANCE  Rigid-body stance check on the GS3DX sole-contact law (#11709).
%
%   REPORT = GS3DX_RIGID_STANCE(INFO) builds a NEW, unsaved Simscape model
%   (never saved to disk, and never touching GS3DX_FullBodyContact.slx)
%   with ONE rigid body -- the GS3DX_FullBodyContact golfer's total mass at
%   its world centre of mass at the stance start pose -- resting on the
%   SAME six sole-sphere contacts (geometry and Spatial Contact Force law
%   and parameters, by workspace variable name and value) as
%   GS3DX_FullBodyContact, and simulates it to rest.
%
%   Why a rigid proxy (director-approved option (b), #11709 2026-10-10
%   comment): the full articulated model costs ~140 s of wall clock per
%   simulated ms once every joint is locked (GS3DX_LOCK_JOINTS), so a
%   full-model stance run is infeasible as a routine cross-engine-parity
%   check. The rigid body tests exactly what the stance parity compares --
%   the contact law under the body's weight and centre of mass -- without
%   paying for the full articulated tree: one 6-DOF joint and six
%   zero-DOF contact points compile to a few dozen blocks, not the
%   967-block GS3DX_FullBodyContact.
%
%   Mass, COM and the per-foot ankle pose come from a SHORT
%   (OPTS.probe_time, default 4 ms) run of GS3DX_FullBodyContact at rest
%   (GS3DX_CONTACT_CHECK, rest=true): the same technique
%   GS3DX_BUILD_CONTACT already uses to read the pelvis pose at t=0, and
%   cheap because it reads the contact model's own start pose rather than
%   a settled trajectory -- the expensive part of the full model is
%   integrating the stiff contact dynamics over time, not evaluating its
%   forward kinematics at t=0. The six sole-sphere world centres then come
%   from GS3DX_SOLE_CONTACT_POINTS, the same placement
%   EXPORT_GS3DX_GRF_FIXTURE uses for its receipt, so the rigid proxy
%   starts at the same ground-penetration state (and hence feels the same
%   initial contact force) as the full model does at t=0.
%
%   The new body's rotational inertia is a solid sphere of the same total
%   mass with radius OPTS.inertia_radius (default 0.15 m, a torso-scale
%   approximation: no measurement backs it). A settled standing stance
%   carries no net torque about the COM, so the forces this check reports
%   at rest do not depend on the inertia tensor -- only the transient
%   settling trajectory and its duration do. This is a documented
%   approximation, not a measurement (see GROUND_CONTACT.md).
%
%   REPORT fields:
%     .mass_kg          total mass from the probe run (kg)
%     .com_m            world COM at the probe pose (3x1, m)
%     .sphere_points_m  world sphere centres at the probe pose, in
%                       GS3DX_SOLE_CONTACT_POINTS column order (3x6, m)
%     .geo              GS3DX_CONTACT_GEOMETRY struct of the source model
%     .t                output times (1xN, s)
%     .forces_n         per-contact ground-on-body force, world axes,
%                       GS3DX_SOLE_CONTACT_POINTS column order (3x6xN, N)
%     .com_p            rigid-body centre-of-mass position, world (3xN, m)
%     .budget           GS3DX_BLOCK_BUDGET of the rigid rig (compiled=true)
%     .wall_s           simulation wall-clock seconds
%
%   Options: probe_time (0.004 s), stop_time (2.0 s), solver ('ode23t'),
%   rel_tol (1e-6), inertia_radius (0.15 m), output_dt (1e-3 s).
%
%   Preconditions: INFO is GS3DX_SETUP's struct.
%   Postconditions: the rigid rig's compiled block count is at or under
%   GS3DX_NAMES().license_block_limit (asserted); GS3DX_FullBodyContact.slx
%   on disk is unchanged; the rig is closed, unsaved, before returning.

    arguments
        info (1,1) struct %#ok<INUSA> kept for the common GS3DX tool signature
        opts.probe_time (1,1) double {mustBePositive} = 0.004
        opts.stop_time (1,1) double {mustBePositive} = 2.0
        opts.solver (1,:) char {mustBeMember(opts.solver, {'ode23t', 'ode15s', 'daessc'})} = 'ode23t'
        opts.rel_tol (1,1) double {mustBePositive} = 1e-6
        opts.inertia_radius (1,1) double {mustBePositive} = 0.15
        opts.output_dt (1,1) double {mustBePositive} = 1e-3
    end
    mdl = char(gs3dx_names().variants.contact);

    probe = gs3dx_contact_check(info, rest=true, stop_time=opts.probe_time);
    assert(isfield(probe.feet.L, 'R') && isfield(probe.feet.R, 'R'), 'gs3dx:rigid_stance', ...
        'Precondition: AnkleLogs carries no Rotation_Transform; cannot place the sole spheres');
    geo = gs3dx_contact_geometry(mdl);
    feet = struct( ...
        'L', struct('p', probe.feet.L.p(:, 1), 'R', probe.feet.L.R(:, :, 1)), ...
        'R', struct('p', probe.feet.R.p(:, 1), 'R', probe.feet.R.R(:, :, 1)));
    centres = gs3dx_sole_contact_points(geo, feet);
    centres = centres(:, :, 1);
    mass = probe.mass;
    com = probe.com(:, 1);

    rig = 'gs3dx_rigid_stance_rig';
    if bdIsLoaded(rig)
        close_system(rig, 0);
    end
    new_system(rig);
    cleanup = onCleanup(@() close_system(rig, 0));
    mw = get_param(rig, 'ModelWorkspace');
    param_names = ["FootContactStiffness", "FootContactDamping", "FootContactTransitionWidth", ...
                   "FootStaticFriction", "FootDynamicFriction", "FootFrictionCriticalVelocity", ...
                   "FootContactRadius"];
    for n = param_names
        assignin(mw, char(n), geo.(n));
    end

    add = @(lib, name, pos) add_block(lib, [rig '/' name], 'Position', pos);
    world  = add('sm_lib/Frames and Transforms/World Frame', 'World', [20 400 60 440]);
    mech   = add('sm_lib/Utilities/Mechanism Configuration', 'Mechanism', [20 500 60 540]);
    solver = add('nesl_utility/Solver Configuration', 'Solver', [20 600 60 640]);
    set_param(mech, 'GravityVector', '[0 0 -9.80665]');

    root = add('sm_lib/Joints/6-DOF Joint', 'Root Joint', [160 380 220 440]);
    offset = add('sm_lib/Frames and Transforms/Rigid Transform', 'COM Offset', [320 380 360 420]);
    set_param(offset, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffsetUnits', 'm', ...
        'TranslationCartesianOffset', mat2str(com.', 17));
    body = add('sm_lib/Body Elements/Spherical Solid', 'Body', [420 380 460 420]);
    set_param(body, 'SphereRadius', num2str(opts.inertia_radius, 17), 'SphereRadiusUnits', 'm', ...
        'BasedOnType', 'Mass', 'Mass', num2str(mass, 17), 'MassUnits', 'kg');
    % Sensed at the COM Offset frame (the Spherical Solid's own reference
    % frame, i.e. its centre), not the joint's base frame, so this reports
    % the true world centre of mass even if the body rotates.
    sensor = add('sm_lib/Frames and Transforms/Transform Sensor', 'COM Sensor', [320 300 380 350]);
    set_param(sensor, 'MeasurementFrame', 'World', 'SenseXYZ', 'on');
    conv_p = add('nesl_utility/PS-Simulink Converter', 'COM Position Converter', [440 300 470 330]);
    set_param(conv_p, 'Unit', 'm');
    term_p = add('simulink/Sinks/Terminator', 'COMPosition', [500 300 520 320]);

    wp = local_port(world, 'RConn', 1);
    rp = get_param(root, 'PortHandles');
    local_connect(rig, wp, rp.LConn(1));
    local_connect(rig, wp, local_port(mech, 'RConn', 1));
    local_connect(rig, wp, local_port(solver, 'RConn', 1));
    local_connect(rig, rp.RConn(1), local_port(offset, 'LConn', 1));
    com_frame = local_port(offset, 'RConn', 1);
    local_connect(rig, com_frame, local_port(body, 'RConn', 1));
    sp = get_param(sensor, 'PortHandles');
    local_connect(rig, wp, sp.LConn(1));
    local_connect(rig, com_frame, sp.RConn(1));
    local_connect(rig, sp.RConn(2), local_port(conv_p, 'LConn', 1));
    h = add_line(rig, local_port(conv_p, 'Outport', 1), local_port(term_p, 'Inport', 1), 'autorouting', 'on');
    set_param(h, 'Name', 'COMPosition');
    set_param(local_port(conv_p, 'Outport', 1), 'DataLogging', 'on');

    frame = add('sm_lib/Frames and Transforms/Rigid Transform', 'Ground Frame', [160 700 200 740]);
    set_param(frame, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffsetUnits', 'm', ...
        'TranslationCartesianOffset', mat2str(geo.GroundOffset.', 17), ...
        'RotationMethod', 'RotationMatrix', 'RotationMatrix', mat2str(geo.GroundRotation, 17));
    plane = add('sm_lib/Curves and Surfaces/Infinite Plane', 'Ground Plane', [260 700 300 740]);
    local_connect(rig, wp, local_port(frame, 'LConn', 1));
    local_connect(rig, local_port(frame, 'RConn', 1), local_port(plane, 'LConn', 1));
    ground = local_port(plane, 'RConn', 1);

    labels = ["LHeel", "LToeIn", "LToeOut", "RHeel", "RToeIn", "RToeOut"];
    forces = zeros(1, 6);
    for k = 1:6
        yy = 100 + 90 * k;
        name = char(labels(k));
        rt = add('sm_lib/Frames and Transforms/Rigid Transform', [name ' Point'], [540 yy 580 yy + 40]);
        set_param(rt, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffsetUnits', 'm', ...
            'TranslationCartesianOffset', mat2str(centres(:, k).', 17));
        ball = add('sm_lib/Body Elements/Spherical Solid', [name ' Sphere'], [640 yy 680 yy + 40]);
        set_param(ball, 'SphereRadius', 'FootContactRadius', 'SphereRadiusUnits', 'm', ...
            'BasedOnType', 'Mass', 'Mass', '1e-3', 'MassUnits', 'kg', 'ExportEntireGeometry', 'on');
        cf = add('sm_lib/Forces and Torques/Spatial Contact Force', [name ' Contact'], [760 yy 820 yy + 40]);
        set_param(cf, 'NormalStiffness', 'FootContactStiffness', 'NormalDamping', 'FootContactDamping', ...
            'NormalTransitionRegionWidth', 'FootContactTransitionWidth', ...
            'FrictionType', 'SmoothStickSlip', 'CoefficientOfStaticFriction', 'FootStaticFriction', ...
            'CoefficientOfDynamicFriction', 'FootDynamicFriction', ...
            'FrictionalCriticalVelocity', 'FootFrictionCriticalVelocity', ...
            'SenseTotalForce', 'on', 'SensingForceTorqueDirection', 'BaseOnFollower', ...
            'SensingForceTorqueResolutionFrame', 'BaseFrame');
        local_connect(rig, rp.RConn(1), local_port(rt, 'LConn', 1));
        local_connect(rig, local_port(rt, 'RConn', 1), local_port(ball, 'RConn', 1));
        local_connect(rig, ground, local_port(cf, 'LConn', 1));
        local_connect(rig, local_port(cf, 'RConn', 1), local_port(ball, 'LConn', 1));
        forces(k) = local_port(cf, 'RConn', 2);
    end
    gs3dx_log_force_mux(rig, forces, 'ContactForces');

    budget = gs3dx_block_budget(rig, compiled=true);
    assert(budget.compiled_total <= gs3dx_names().license_block_limit, 'gs3dx:rigid_stance', ...
        'Postcondition: the rigid stance rig compiles to %d blocks, over the %d license limit', ...
        budget.compiled_total, gs3dx_names().license_block_limit);

    n_out = round(opts.stop_time / opts.output_dt) + 1;
    output_times = (0:n_out - 1).' * opts.output_dt;
    set_param(rig, 'StopTime', num2str(opts.stop_time, 17), 'SolverType', 'Variable-step', ...
        'Solver', opts.solver, 'RelTol', num2str(opts.rel_tol, 17), 'AbsTol', num2str(opts.rel_tol, 17), ...
        'OutputOption', 'SpecifiedOutputTimes', 'OutputTimes', mat2str(output_times, 17), ...
        'ReturnWorkspaceOutputs', 'on', 'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
    tic;
    sim_out = sim(rig);
    wall_s = toc;
    assert(isempty(sim_out.ErrorMessage), 'gs3dx:rigid_stance', '%s', sim_out.ErrorMessage);

    f = sim_out.logsout.get('ContactForces').Values.Data;
    f = reshape(f, 3, 6, []);
    p = sim_out.logsout.get('COMPosition').Values.Data;
    p = reshape(p, 3, []);

    report.mass_kg = mass;
    report.com_m = com;
    report.sphere_points_m = centres;
    report.geo = geo;
    report.t = output_times.';
    report.forces_n = f;
    report.com_p = p;
    report.budget = budget;
    report.wall_s = wall_s;
end

function local_connect(sys, a, b)
    add_line(sys, a, b, 'autorouting', 'on');
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end
