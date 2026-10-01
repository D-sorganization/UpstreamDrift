classdef test_gs3dx_feedback_torque < matlab.unittest.TestCase
%TEST_GS3DX_FEEDBACK_TORQUE  The feedback torque a tracked run still needs (#11173).
%
%   GS3DX_FEEDBACK_TORQUE measures the distance from a servo-tracked run to
%   pure forward dynamics (docs/FORWARD_DYNAMICS.md): every torque that is
%   not the feedforward.  Synthetic references and states only; nothing
%   is simulated.

    methods (TestClassSetup)
        function setup(~)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
        end
    end

    methods (Test)
        function no_error_needs_no_feedback(testCase)
            [p, state] = local_data(0, 0);
            fb = gs3dx_feedback_torque(p, state);
            testCase.verifyEqual(fb.total_rms_Nm, 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(height(fb.joints), 2 + 1 + 12 + 12, 'Spine XY, Torso, 12 leg axes, 12 balance');
            testCase.verifyEqual(fb.joints.rms_Nm, zeros(height(fb.joints), 1), 'AbsTol', 1e-12);
        end

        function an_angle_error_costs_kp_times_the_error(testCase)
            [p, state] = local_data(2, 0);   % every angle 2 deg short of its reference
            fb = gs3dx_feedback_torque(p, state);
            spine = fb.joints(fb.joints.joint == "Spine", :);
            testCase.verifyEqual(spine.rms_Nm, 2 * p.SpineTrackKp(:), 'RelTol', 1e-12);
            legs = fb.joints(fb.joints.group == "legs", :);
            testCase.verifyEqual(legs.rms_Nm, 2 * p.LegServoKp(:), 'RelTol', 1e-12);
            testCase.verifyEqual(legs.peak_Nm, legs.rms_Nm, 'RelTol', 1e-12, 'constant error');
            all_u = [2 * p.SpineTrackKp(:); 2 * p.TorsoTrackKp; 2 * p.LegServoKp(:)];
            testCase.verifyEqual(fb.total_rms_Nm, sqrt(mean(all_u .^ 2)), 'RelTol', 1e-12);
        end

        function balance_share_is_the_command_change(testCase)
            [p, state] = local_data(0, 0.02);   % COM 2 cm off its reference
            fb = gs3dx_feedback_torque(p, state);
            T = p.LegReferenceTime;
            k = 3;
            args = {T, p.LegTorqueCommand, p.LegServoKp, p.LegServoKd, p.LegReferenceAngle, p.LegReferenceRate, ...
                p.BalanceCOMRef, p.BalanceCOMRate, p.BalanceFootRef, p.BalanceGain, p.BalanceKp, p.BalanceKd, ...
                p.BalanceFootKp, p.BalanceLimit};
            on = gs3dx_balance_command(state.t(k), state.com(:, k), state.com_rate(:, k), state.feet(:, k), args{:}, 1);
            off = gs3dx_balance_command(state.t(k), state.com(:, k), state.com_rate(:, k), state.feet(:, k), args{:}, 0);
            testCase.verifyEqual(fb.torque.balance(:, k), on - off, 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(max(fb.joints.rms_Nm(fb.joints.group == "balance")), 0);
            p.BalanceOn = 0;
            fb = gs3dx_feedback_torque(p, state);
            testCase.verifyEqual(fb.torque.balance, zeros(12, numel(state.t)));
        end

        function a_model_without_balance_reports_legs_only(testCase)
            [p, state] = local_data(1, 0);
            p = rmfield(p, {'BalanceCOMRef', 'BalanceCOMRate', 'BalanceFootRef', 'BalanceGain', 'BalanceKp', ...
                'BalanceKd', 'BalanceFootKp', 'BalanceLimit', 'BalanceOn'});
            state = rmfield(state, {'com', 'com_rate', 'feet'});
            fb = gs3dx_feedback_torque(p, state);
            testCase.verifyFalse(any(fb.joints.group == "balance"));
            testCase.verifyEqual(fb.groups.group, ["upper"; "legs"]);
        end

        function a_state_of_the_wrong_size_is_refused(testCase)
            [p, state] = local_data(0, 0);
            state.legs.q = state.legs.q(1:6, :);
            testCase.verifyError(@() gs3dx_feedback_torque(p, state), 'gs3dx:feedback');
            [p, state] = local_data(0, 0);
            state.upper.Spine.qd = state.upper.Spine.qd(:, 1:end - 1);
            testCase.verifyError(@() gs3dx_feedback_torque(p, state), 'gs3dx:feedback');
        end
    end
end

function [p, state] = local_data(err_deg, com_err)
% Two upper-body charts (Spine XY, Torso Rz), both legs and a balance loop,
% on 5 reference frames; the state lags every angle by ERR_DEG and the
% centre of mass sits COM_ERR (m) off its reference along World x.
    T = linspace(0, 0.4, 5);
    n = numel(T);
    ramp = @(rows, scale) scale * (1:rows).' * T;
    p.UpperBodyTrackTime = T;
    p.SpineTrackAngle = ramp(2, 10);  p.SpineTrackRate = repmat(10 * (1:2).', 1, n);
    p.SpineTrackKp = [40 30];         p.SpineTrackKd = [1 1];
    p.TorsoTrackAngle = ramp(1, 20);  p.TorsoTrackRate = repmat(20, 1, n);
    p.TorsoTrackKp = 50;              p.TorsoTrackKd = 2;
    p.LegReferenceTime = T;
    p.LegReferenceAngle = ramp(12, 3); p.LegReferenceRate = repmat(3 * (1:12).', 1, n);
    p.LegServoKp = repmat([100 100 100 100 50 50], 1, 2);
    p.LegServoKd = repmat([2 2 2 2 1 1], 1, 2);
    p.LegTorqueCommand = zeros(12, 1);
    p.BalanceCOMRef = repmat([0; 0; 1], 1, n);  p.BalanceCOMRate = zeros(3, n);
    p.BalanceFootRef = repmat([0; 0.2; 0; 0; -0.2; 0], 1, n);
    p.BalanceGain = repmat(reshape(1:36, 12, 3) / 10, 1, 1, n);
    p.BalanceKp = 3;  p.BalanceKd = 0.4;  p.BalanceFootKp = 1;  p.BalanceLimit = 0.1;  p.BalanceOn = 1;

    state.t = T;
    state.upper.Spine = struct('q', p.SpineTrackAngle - err_deg, 'qd', p.SpineTrackRate);
    state.upper.Torso = struct('q', p.TorsoTrackAngle - err_deg, 'qd', p.TorsoTrackRate);
    state.legs = struct('q', p.LegReferenceAngle - err_deg, 'qd', p.LegReferenceRate);
    state.com = p.BalanceCOMRef + [com_err; 0; 0];
    state.com_rate = zeros(3, n);
    state.feet = p.BalanceFootRef;
end
