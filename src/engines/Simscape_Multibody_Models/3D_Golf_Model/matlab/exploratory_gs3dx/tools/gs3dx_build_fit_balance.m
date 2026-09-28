function report = gs3dx_build_fit_balance(info, ref, opts)
%GS3DX_BUILD_FIT_BALANCE  Build GS3DX_FitBalance: GS3DX_FitTrack with centre-of-mass and foot feedback (#10979).
%
%   REPORT = GS3DX_BUILD_FIT_BALANCE(INFO, REF) copies GS3DX_FitTrack to
%   GS3DX_FitBalance and closes a balance loop through the leg servo.  REF
%   is the leg reference (GS3DX_LEG_REFERENCE) GS3DX_FitLegs was built from.
%
%   * Sensing.  A whole-mechanism Inertia Sensor measures the centre of mass
%     in World; a PS-Simulink Converter and a global Goto (GS3DXBalanceCOM)
%     carry it into 'Lower Body', where a State-Space block differentiates
%     it (first-order filter, BalanceCOMTau).  Bus Selectors take each
%     ankle's GlobalPosition from its <L|R>AnkleLogs bus.
%   * Command.  'Lower Body/Leg Torque Commands' becomes a MATLAB Function
%     of the same name calling GS3DX_BALANCE_COMMAND on a Clock: the
%     GS3DX_FitLegs servo command plus the leg angle offsets
%     (GS3DX_BALANCE_GAIN) that shift the pelvis over the feet against the
%     centre-of-mass error, and that move each foot back toward its
%     reference (BalanceFootKp).  With BalanceOn = 0 it plays exactly the
%     GS3DX_FitLegs command.
%   * Reference.  The centre-of-mass reference is the reference pelvis pose
%     carrying COM_OFFSET (3 x frames, the centre of mass in the pelvis
%     frame, m; GS3DX_BALANCE_COM_OFFSET of a run whose joints track), or
%     COM_REF (3 x frames, World, m) given directly, such as the capture's
%     own centre of mass (GS3DX_CAPTURE_COM_REFERENCE).  Without either,
%     BalanceOn is 0.  The foot reference is REF.feet.
%
%   Options: overwrite (false), com_offset ([]), com_ref ([]), gains ([BalanceKp BalanceKd],
%   [3 0.4]: m/m and s; a pelvis shift moves the whole-body centre of mass
%   by only part of the shift, so Kp 1 left a standing error, docs/FIT.md),
%   limit (0.1 m, the largest pelvis shift), tau
%   (0.01 s), axes (3: the error along World x, y and z; 2: horizontal
%   only), foot_gain (1 m/m, BalanceFootKp).  REPORT fields: .on, .gains,
%   .budget.

    arguments
        info (1,1) struct
        ref (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.com_offset double = []
        opts.com_ref double = []
        opts.gains (1,2) double {mustBeNonnegative} = [3 0.4]
        opts.limit (1,1) double {mustBePositive} = 0.1
        opts.tau (1,1) double {mustBePositive} = 0.01
        opts.axes (1,1) double {mustBeMember(opts.axes, [2 3])} = 3
        opts.foot_gain (1,1) double {mustBeNonnegative} = 1
    end
    assert(isempty(opts.com_offset) || isempty(opts.com_ref), 'gs3dx:fitbalance', ...
        'Give COM_OFFSET or COM_REF, not both');
    names = gs3dx_names();
    src = char(names.variants.fit_track);
    mdl = char(names.variants.fit_balance);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:fitbalance');

    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    T = ws.getVariable('LegReferenceTime');
    n = numel(T);
    A = ws.getVariable('LegReferenceAngle');
    assert(numel(ref.frames) == n && isequal(size(ref.q), size(A)) && max(abs(ref.q - A), [], 'all') < 1e-9, ...
        'gs3dx:fitbalance', ...
        'REF is not the leg reference of %s', src);

    report.on = gs3dx_balance_reference(ws, ref, opts.com_offset, opts.com_ref, 'gs3dx:fitbalance');
    assignin(ws, 'BalanceKp', opts.gains(1));
    assignin(ws, 'BalanceKd', opts.gains(2));
    assignin(ws, 'BalanceLimit', opts.limit);
    assignin(ws, 'BalanceCOMTau', opts.tau);
    assignin(ws, 'BalanceGain', gs3dx_balance_gain(ref, ws, axes=opts.axes));
    assignin(ws, 'BalanceFootKp', opts.foot_gain);
    assignin(ws, 'BalanceFootRef', [ref.feet.L.p; ref.feet.R.p]);
    report.gains = [opts.gains opts.foot_gain];

    local_com_sensor(mdl);
    local_command(mdl);
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % GS3DX_CONTACT_CHECK's sensors compile to 10 more (973 -> 983, 2026-09-28)
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:fitbalance', ...
        '%s compiles to %d blocks; %d leave no room for %d validation blocks', ...
        mdl, report.budget.compiled_total, names.license_block_limit, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
end

function local_com_sensor(mdl)
% Whole-mechanism centre of mass in World, to a global Goto.
    world = gs3dx_pm_port([mdl '/Hips and Torso Inputs'], 'GlobalReferenceFrame');
    s = add_block('sm_lib/Body Elements/Inertia Sensor', [mdl '/GS3DX Balance COM Sensor'], ...
        'SensorExtent', 'Mechanism', 'SenseMass', 'off', 'SenseCenterOfMass', 'on', ...
        'SenseInertiaMatrix', 'off', 'Position', [100 2800 160 2850]);
    sp = get_param(s, 'PortHandles');
    assert(isscalar(sp.RConn), 'gs3dx:fitbalance', 'Expected one Inertia Sensor output');
    c = add_block('nesl_utility/PS-Simulink Converter', [mdl '/GS3DX Balance COM Converter'], ...
        'Unit', 'm', 'Position', [220 2810 250 2840]);
    g = add_block('simulink/Signal Routing/Goto', [mdl '/GS3DX Balance COM Goto'], ...
        'GotoTag', 'GS3DXBalanceCOM', 'TagVisibility', 'global', 'Position', [300 2810 380 2840]);
    cp = get_param(c, 'PortHandles');
    gp = get_param(g, 'PortHandles');
    add_line(mdl, world, sp.LConn(1));
    add_line(mdl, sp.RConn(1), cp.LConn(1));
    add_line(mdl, cp.Outport(1), gp.Inport(1));
end

function local_command(mdl)
% Replace the servo From Workspace with a MATLAB Function of the same name.
    sys = [mdl '/Lower Body'];
    blk = [sys '/Leg Torque Commands'];
    assert(strcmp(get_param(blk, 'BlockType'), 'FromWorkspace'), 'gs3dx:fitbalance', ...
        '%s is not the GS3DX_FitLegs servo From Workspace', blk);
    lines = get_param(blk, 'LineHandles');
    dst = get_param(lines.Outport, 'DstPortHandle');
    pos = get_param(blk, 'Position');
    delete_line(lines.Outport);
    delete_block(blk);

    add_block('simulink/User-Defined Functions/MATLAB Function', blk, 'Position', pos);
    chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', blk);
    params = {'LegReferenceTime', 'LegTorqueCommand', 'LegServoKp', 'LegServoKd', 'LegReferenceAngle', ...
        'LegReferenceRate', 'BalanceCOMRef', 'BalanceCOMRate', 'BalanceFootRef', 'BalanceGain', 'BalanceKp', ...
        'BalanceKd', 'BalanceFootKp', 'BalanceLimit', 'BalanceOn'};
    args = strjoin(params, ', ');
    chart.Script = sprintf(['function cmd = fcn(t, com, com_rate, feet, %s)\n' ...
        '%% Leg servo command with centre-of-mass and foot feedback (GS3DX_BUILD_FIT_BALANCE, #10979).\n' ...
        'cmd = gs3dx_balance_command(t, com, com_rate, feet, %s);\n'], args, args);
    for p = params
        d = chart.find('-isa', 'Stateflow.Data', 'Name', p{1});
        assert(isscalar(d), 'gs3dx:fitbalance', 'No data %s in %s', p{1}, blk);
        d.Scope = 'Parameter';
    end

    x = pos(1) - 300;
    y = pos(2);
    clk = add_block('simulink/Sources/Clock', [sys '/Balance Clock'], 'Position', [x y x + 20 y + 20]);
    from = add_block('simulink/Signal Routing/From', [sys '/Balance COM'], 'GotoTag', 'GS3DXBalanceCOM', ...
        'Position', [x y + 40 x + 80 y + 60]);
    rate = add_block('simulink/Continuous/State-Space', [sys '/Balance COM Rate'], ...
        'A', '-eye(3) / BalanceCOMTau', 'B', 'eye(3) / BalanceCOMTau', ...
        'C', '-eye(3) / BalanceCOMTau', 'D', 'eye(3) / BalanceCOMTau', ...
        'InitialCondition', 'BalanceCOMRef(:, 1)', 'Position', [x + 120 y + 80 x + 200 y + 110]);
    fp = get_param(blk, 'PortHandles');
    port = @(b, k) local_outport(b, k);
    add_line(sys, port(clk, 1), fp.Inport(1), 'autorouting', 'on');
    add_line(sys, port(from, 1), fp.Inport(2), 'autorouting', 'on');
    rp = get_param(rate, 'PortHandles');
    add_line(sys, port(from, 1), rp.Inport(1), 'autorouting', 'on');
    add_line(sys, rp.Outport(1), fp.Inport(3), 'autorouting', 'on');
    mux = add_block('simulink/Signal Routing/Mux', [sys '/Balance Feet'], 'Inputs', '2', ...
        'Position', [x + 120 y + 140 x + 125 y + 200]);
    mp = get_param(mux, 'PortHandles');
    sides = 'LR';
    for s = 1:2
        logs = find_system(sys, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line', 'Name', [sides(s) 'AnkleLogs']);
        assert(~isempty(logs), 'gs3dx:fitbalance', 'No %sAnkleLogs bus in %s', sides(s), sys);
        sel = add_block('simulink/Signal Routing/Bus Selector', [sys '/Balance ' sides(s) ' Ankle'], ...
            'Position', [x y + 130 + 40 * s x + 10 y + 160 + 40 * s]);
        add_line(sys, get_param(logs(1), 'SrcPortHandle'), local_inport(sel), 'autorouting', 'on');
        set_param(sel, 'OutputSignals', 'GlobalPosition');
        add_line(sys, port(sel, 1), mp.Inport(s), 'autorouting', 'on');
    end
    add_line(sys, mp.Outport(1), fp.Inport(4), 'autorouting', 'on');
    for d = dst(:).'
        add_line(sys, fp.Outport(1), d, 'autorouting', 'on');
    end
end

function h = local_inport(blk)
    ph = get_param(blk, 'PortHandles');
    h = ph.Inport(1);
end

function h = local_outport(blk, k)
    ph = get_param(blk, 'PortHandles');
    h = ph.Outport(k);
end
