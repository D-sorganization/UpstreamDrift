function trial = gs3dx_contact_trial(info, opts)
%GS3DX_CONTACT_TRIAL  Cost and speed of foot-ground contact in place of the weld.
%
%   TRIAL = GS3DX_CONTACT_TRIAL(INFO) loads GS3DX_FullBody, replaces each
%   'Foot Ground' weld frame with a Spatial Contact Force between the foot
%   brick and one Infinite Plane at sole height, and simulates the drive
%   (#10958).  The model is closed WITHOUT saving: this measures the
%   alternative, it does not adopt it.  The legs carry no torque, so the
%   contact run only prices the blocks and the solver cost; its motion is
%   not meaningful.
%
%   TRIAL fields: .blocks_added, .blocks_removed (non-virtual), .weld and
%   .contact (GS3DX_SIMULATE results for the same drive and stop time;
%   wall_s is the fastest of opts.repeats runs, so compilation is excluded).
%
%   Options: drive ("impact"), stop_time (0.3 s), stiffness (N/m),
%   damping (N/(m/s)), repeats (3).

    arguments
        info (1,1) struct
        opts.drive (1,1) string = "impact"
        opts.stop_time (1,1) double {mustBePositive} = 0.3
        opts.stiffness (1,1) double {mustBePositive} = 1e6
        opts.damping (1,1) double {mustBePositive} = 1e3
        opts.repeats (1,1) double {mustBeInteger, mustBePositive} = 3
    end
    names = gs3dx_names();
    full = char(names.variants.fullbody);
    load_system(full);
    cleanup = onCleanup(@() close_system(full, 0));
    vars = gs3dx_drive(info, opts.drive, full);
    trial.weld = local_timed(full, vars, opts);

    sys = [full '/Lower Body'];
    world = local_conn_port([sys '/World']);
    ws = get_param(full, 'ModelWorkspace');
    sole = ws.getVariable('LFootGroundOffset') - ws.getVariable('AnkleHeight') * ws.getVariable('LFootGroundRotation') * [0; 0; 1];
    plane_frame = add_block('sm_lib/Frames and Transforms/Rigid Transform', [sys '/Ground Plane Frame'], ...
        'RotationMethod', 'RotationMatrix', 'RotationMatrix', 'LFootGroundRotation', ...
        'TranslationMethod', 'Cartesian', 'TranslationCartesianOffset', mat2str(sole.', 17), ...
        'TranslationCartesianOffsetUnits', 'm', 'Position', [1500 900 1550 950]);
    plane = add_block('sm_lib/Curves and Surfaces/Infinite Plane', [sys '/Ground Plane'], ...
        'Position', [1620 900 1680 950]);
    add_line(sys, world, local_port(plane_frame, 'LConn', 1), 'autorouting', 'on');
    % Infinite Plane: frame R on the left, geometry G on the right.
    add_line(sys, local_port(plane_frame, 'RConn', 1), local_port(plane, 'LConn', 1), 'autorouting', 'on');
    plane_geometry = local_port(plane, 'RConn', 1);
    trial.blocks_added = 2;
    trial.blocks_removed = 0;
    for side = 'LR'
        ground = [sys '/' side ' Foot Ground'];
        lines = get_param(ground, 'LineHandles');
        delete_line([lines.LConn, lines.RConn]);
        delete_block(ground);
        foot = [sys '/' side ' Foot'];
        set_param(foot, 'ExportEntireGeometry', 'on');
        foot_geometry = local_port(foot, 'LConn', 1);   % exported geometry G; frame R stays right
        contact = add_block('sm_lib/Forces and Torques/Spatial Contact Force', [sys '/' side ' Foot Contact'], ...
            'NormalStiffness', num2str(opts.stiffness), 'NormalDamping', num2str(opts.damping), ...
            'Position', [1620 1000 + 100 * (side == 'R') 1680 1050 + 100 * (side == 'R')]);
        add_line(sys, plane_geometry, local_port(contact, 'LConn', 1), 'autorouting', 'on');
        add_line(sys, local_port(contact, 'RConn', 1), foot_geometry, 'autorouting', 'on');
        trial.blocks_added = trial.blocks_added + 1;
        trial.blocks_removed = trial.blocks_removed + 1;
    end
    trial.contact = local_timed(full, vars, opts);
end

function run = local_timed(mdl, vars, opts)
    run = gs3dx_simulate(mdl, variables = vars, stop_time = opts.stop_time);
    for k = 2:opts.repeats
        again = gs3dx_simulate(mdl, variables = vars, stop_time = opts.stop_time);
        run.wall_s = min(run.wall_s, again.wall_s);
    end
end

function p = local_conn_port(blk)
    ph = get_param(blk, 'PortHandles');
    p = [ph.LConn, ph.RConn];
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end
