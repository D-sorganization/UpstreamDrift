function gs3dx_rig_scaffold(rig, base, follower)
%GS3DX_RIG_SCAFFOLD  World, gravity, solver and a test body for a joint rig.
%
%   GS3DX_RIG_SCAFFOLD(RIG, BASE, FOLLOWER) adds to the new, unsaved model
%   RIG a World frame, a Mechanism Configuration with gravity -Z and a
%   Solver Configuration, connects the World to physical port BASE, and
%   hangs a 1.8 kg brick at an offset of [0.25 0.08 -0.05] m on physical
%   port FOLLOWER, so gravity and the drive load every rotation axis.

    arguments
        rig (1,:) char
        base (1,1) double
        follower (1,1) double
    end
    add = @(lib, name, pos) add_block(lib, [rig '/' name], 'Position', pos);
    world  = add('sm_lib/Frames and Transforms/World Frame', 'World', [20 200 60 240]);
    mech   = add('sm_lib/Utilities/Mechanism Configuration', 'Mechanism', [20 300 60 340]);
    solver = add('nesl_utility/Solver Configuration', 'Solver', [20 400 60 440]);
    offset = add('sm_lib/Frames and Transforms/Rigid Transform', 'Offset', [360 200 400 240]);
    body   = add('sm_lib/Body Elements/Brick Solid', 'Body', [460 200 500 240]);
    set_param(offset, 'TranslationMethod', 'Cartesian', 'TranslationCartesianOffset', '[0.25 0.08 -0.05]');
    set_param(body, 'BrickDimensions', '[0.3 0.1 0.06]');   % 1.8 kg at the default density
    set_param(mech, 'GravityVector', '[0 0 -9.80665]');
    w = local_port(world, 'RConn', 1);
    add_line(rig, w, base, 'autorouting', 'on');
    add_line(rig, w, local_port(mech, 'RConn', 1), 'autorouting', 'on');
    add_line(rig, w, local_port(solver, 'RConn', 1), 'autorouting', 'on');
    add_line(rig, follower, local_port(offset, 'LConn', 1), 'autorouting', 'on');
    add_line(rig, local_port(offset, 'RConn', 1), local_port(body, 'RConn', 1), 'autorouting', 'on');
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end
