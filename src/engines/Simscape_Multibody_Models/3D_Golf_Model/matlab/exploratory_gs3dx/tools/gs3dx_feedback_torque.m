function fb = gs3dx_feedback_torque(p, state)
%GS3DX_FEEDBACK_TORQUE  Feedback torque of a tracked run: its distance to pure forward dynamics (#11173).
%
%   FB = GS3DX_FEEDBACK_TORQUE(P, STATE) is every joint torque a run of a
%   servo-tracked GS3DX variant applied beyond its feedforward, on the
%   times STATE.t (s).  docs/FORWARD_DYNAMICS.md drives this to zero.
%
%   P is a struct of the model-workspace variables the run used:
%     upper body  UpperBodyTrackTime and, per chart prefix <P>
%                 (GS3DX_UPPER_BODY_JOINTS), <P>TrackAngle, <P>TrackRate
%                 (axes x frames, deg, deg/s), <P>TrackKp, <P>TrackKd;
%     legs        LegReferenceTime, LegReferenceAngle, LegReferenceRate
%                 (12 x frames), LegServoKp, LegServoKd, LegTorqueCommand;
%     balance     (optional) the GS3DX_BALANCE_COMMAND parameters
%                 BalanceCOMRef ... BalanceOn.
%   STATE has .t, .upper.<P>.q/.qd (axes x numel(t), deg, deg/s) for the
%   charts to measure, .legs.q/.qd (12 x numel(t)) in the leg servo's own
%   angles, and with a balance loop .com, .com_rate (3 x numel(t), World,
%   m, m/s) and .feet (6 x numel(t), ankle positions).
%
%   The feedback of an upper-body axis is its PD torque
%   Kp (A(t) - q) + Kd (R(t) - qd).  A leg axis has the servo PD torque
%   toward its reference (group "legs") plus the balance loop's change of
%   the command (group "balance": the command with BalanceOn minus the
%   command with the loop off, both from GS3DX_BALANCE_COMMAND).
%
%   FB fields:
%     .joints   table: group ("upper", "legs", "balance"), joint (prefix
%               or "L hip" ...), axis, rms_Nm and peak_Nm over the run
%     .groups   table: group, rms_Nm (every axis and sample of the group)
%     .total_rms_Nm  RMS over every axis and sample of the total feedback
%               (a leg axis counts its servo and balance torque summed)
%     .torque   struct: .upper.<P> (axes x N), .legs and .balance (12 x N)
%     .t        STATE.t
%   Postcondition: every torque is finite.  The prescribed neck is not
%   included (its torque is computed by Simscape, not commanded).

    t = state.t(:).';
    n = numel(t);
    rows = {};
    all_u = [];
    fb = struct('t', t, 'torque', struct('upper', struct()));

    Tu = local_field(p, 'UpperBodyTrackTime');
    for P = reshape(fieldnames(local_field(state, 'upper')), 1, [])
        pre = P{1};
        x = state.upper.(pre);
        A = local_at(Tu, p.([pre 'TrackAngle']), t);
        R = local_at(Tu, p.([pre 'TrackRate']), t);
        local_size(x.q, size(A), [pre ' angles']);
        local_size(x.qd, size(A), [pre ' rates']);
        u = p.([pre 'TrackKp'])(:) .* (A - x.q) + p.([pre 'TrackKd'])(:) .* (R - x.qd);
        fb.torque.upper.(pre) = u;
        rows = [rows; local_rows("upper", string(pre), local_axes(size(u, 1)), u)]; %#ok<AGROW> 12 charts
        all_u = [all_u; u(:)]; %#ok<AGROW>
    end

    T = p.LegReferenceTime;
    A = local_at(T, p.LegReferenceAngle, t);
    R = local_at(T, p.LegReferenceRate, t);
    local_size(state.legs.q, [12 n], 'leg angles');
    local_size(state.legs.qd, [12 n], 'leg rates');
    servo = p.LegServoKp(:) .* (A - state.legs.q) + p.LegServoKd(:) .* (R - state.legs.qd);
    [joint, axis] = local_leg_axes();
    rows = [rows; local_rows("legs", joint, axis, servo)];
    balance = zeros(12, n);
    if isfield(p, 'BalanceOn')
        args = {T, p.LegTorqueCommand, p.LegServoKp, p.LegServoKd, p.LegReferenceAngle, p.LegReferenceRate, ...
            p.BalanceCOMRef, p.BalanceCOMRate, p.BalanceFootRef, p.BalanceGain, p.BalanceKp, p.BalanceKd, ...
            p.BalanceFootKp, p.BalanceLimit};
        local_size(state.com, [3 n], 'COM');
        local_size(state.com_rate, [3 n], 'COM rate');
        local_size(state.feet, [6 n], 'feet');
        for k = 1:n
            s = {t(k), state.com(:, k), state.com_rate(:, k), state.feet(:, k)};
            balance(:, k) = gs3dx_balance_command(s{:}, args{:}, p.BalanceOn) - ...
                gs3dx_balance_command(s{:}, args{:}, 0);
        end
        rows = [rows; local_rows("balance", joint, axis, balance)];
    end
    fb.torque.legs = servo;
    fb.torque.balance = balance;
    all_u = [all_u; reshape(servo + balance, [], 1)];

    fb.joints = vertcat(rows{:});
    groups = unique(fb.joints.group, 'stable');
    rms_g = zeros(numel(groups), 1);
    for g = 1:numel(groups)
        switch groups(g)
            case "upper"
                u = cellfun(@(f) reshape(fb.torque.upper.(f), [], 1), fieldnames(fb.torque.upper), 'UniformOutput', false);
                u = vertcat(u{:});
            case "legs"
                u = servo(:);
            otherwise
                u = balance(:);
        end
        rms_g(g) = sqrt(mean(u .^ 2));
    end
    fb.groups = table(groups, rms_g, 'VariableNames', {'group', 'rms_Nm'});
    fb.total_rms_Nm = sqrt(mean(all_u .^ 2));
    assert(all(isfinite(all_u)), 'gs3dx:feedback', 'Non-finite feedback torque');
end

function x = local_field(s, name)
    assert(isfield(s, name), 'gs3dx:feedback', 'Missing %s', name);
    x = s.(name);
end

function X = local_at(T, V, t)
% V (rows x numel(T)) at the times t, held outside T (as the charts do).
    assert(size(V, 2) == numel(T), 'gs3dx:feedback', 'A reference has %d samples for %d times', size(V, 2), numel(T));
    tc = min(max(t, T(1)), T(end));
    X = interp1(T(:), V.', tc(:), 'linear').';
end

function local_size(x, sz, what)
    assert(isequal(size(x), sz), 'gs3dx:feedback', '%s are %s, expected %s', what, mat2str(size(x)), mat2str(sz));
end

function ax = local_axes(m)
    names = {"Rz", ["X"; "Y"], ["X"; "Y"; "Z"]};
    ax = names{m};
end

function [joint, axis] = local_leg_axes()
% LegTorqueCommand order: [L hip XYZ, knee, ankle XY, R ...].
    j = ["hip"; "hip"; "hip"; "knee"; "ankle"; "ankle"];
    a = ["X"; "Y"; "Z"; "Rz"; "X"; "Y"];
    joint = ["L " + j; "R " + j];
    axis = [a; a];
end

function r = local_rows(group, joint, axis, u)
    m = size(u, 1);
    joint = repmat(joint, m / numel(joint), 1);
    r = {table(repmat(group, m, 1), joint, axis, sqrt(mean(u .^ 2, 2)), max(abs(u), [], 2), ...
        'VariableNames', {'group', 'joint', 'axis', 'rms_Nm', 'peak_Nm'})};
end
