classdef test_gs3dx_fit_balance < matlab.unittest.TestCase
%TEST_GS3DX_FIT_BALANCE  Centre-of-mass feedback into the leg servo, GS3DX_FitBalance (#10979).
%
%   GS3DX_FitBalance is GS3DX_FitTrack with the leg servo references
%   shifted against the horizontal centre-of-mass error
%   (GS3DX_BUILD_FIT_BALANCE, GS3DX_BALANCE_COMMAND, GS3DX_BALANCE_GAIN).
%   Its centre-of-mass reference comes from a balance-off run to impact
%   (docs/FIT.md), so this test checks the saved model and replays a short
%   stretch rather than rebuilding it.

    properties
        info struct
        mdl char
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.fit_balance);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_FitBalance is built by GS3DX_BUILD_FIT_BALANCE (docs/FIT.md)');
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function balance_off_is_the_leg_servo(testCase)
            [T, C0, Kp, Kd, A, R, Cref, Vref, Fref, G] = local_command_data();
            [cmd, shift] = gs3dx_balance_command(0.5, [0.3; -0.2; 1], [1; 1; 0], ones(6, 1), T, C0, Kp, Kd, ...
                A, R, Cref, Vref, Fref, G, 1, 0.2, 1, 0.1, 0);
            testCase.verifyEqual(shift, [0; 0]);
            testCase.verifyEqual(cmd, C0 + Kp .* (A(:, 1) + A(:, 2)) / 2 + Kd .* (R(:, 1) + R(:, 2)) / 2, ...
                'AbsTol', 1e-12);
        end

        function balance_shifts_against_the_error_up_to_the_limit(testCase)
            [T, C0, Kp, Kd, A, R, Cref, Vref, Fref, G] = local_command_data();
            com = Cref(:, 1) + [0.02; -0.01; 0.5];
            feet = Fref(:, 1);
            [cmd, shift] = gs3dx_balance_command(0, com, Vref(:, 1), feet, T, C0, Kp, Kd, A, R, Cref, Vref, ...
                Fref, G, 1, 0.2, 1, 0.1, 1);
            testCase.verifyEqual(shift, [-0.02; 0.01], 'AbsTol', 1e-12, 'vertical error ignored');
            testCase.verifyEqual(cmd, C0 + Kp .* (A(:, 1) + G(:, :, 1) * shift) + Kd .* R(:, 1), 'AbsTol', 1e-12);
            [~, shift] = gs3dx_balance_command(0, Cref(:, 1) + [3; 4; 0], Vref(:, 1), feet, T, C0, Kp, Kd, ...
                A, R, Cref, Vref, Fref, G, 1, 0.2, 1, 0.1, 1);
            testCase.verifyEqual(shift, -0.1 * [0.6; 0.8], 'AbsTol', 1e-12, 'limited');
            G3 = repmat(reshape(1:36, 12, 3), 1, 1, 2);
            [cmd, shift] = gs3dx_balance_command(0, com, Vref(:, 1), feet, T, C0, Kp, Kd, A, R, Cref, Vref, ...
                Fref, G3, 1, 0.2, 1, 1, 1);
            testCase.verifyEqual(shift, [-0.02; 0.01; -0.5], 'AbsTol', 1e-12, 'three-axis gain uses z');
            testCase.verifyEqual(cmd, C0 + Kp .* (A(:, 1) + G3(:, :, 1) * shift) + Kd .* R(:, 1), 'AbsTol', 1e-12);
        end

        function each_leg_moves_its_foot_back(testCase)
            % A foot away from its reference offsets only its own leg, by
            % the pelvis-shift gain times BKF e_foot, limited per foot.
            [T, C0, Kp, Kd, A, R, Cref, Vref, Fref, ~] = local_command_data();
            G3 = repmat(reshape(1:36, 12, 3), 1, 1, 2);
            ef = [0; 0; 0; 0.01; -0.02; 0.03];
            cmd = gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1) + ef, T, C0, Kp, Kd, A, R, ...
                Cref, Vref, Fref, G3, 0, 0, 2, 1, 1);
            foot = [zeros(6, 1); G3(7:12, :, 1) * (2 * ef(4:6))];
            testCase.verifyEqual(cmd, C0 + Kp .* (A(:, 1) + foot) + Kd .* R(:, 1), 'AbsTol', 1e-12);
            cmd = gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1) + 10 * ef, T, C0, Kp, Kd, A, R, ...
                Cref, Vref, Fref, G3, 0, 0, 2, 0.05, 1);
            foot = [zeros(6, 1); G3(7:12, :, 1) * (0.05 * ef(4:6) / norm(ef(4:6)))];
            testCase.verifyEqual(cmd, C0 + Kp .* (A(:, 1) + foot) + Kd .* R(:, 1), 'AbsTol', 1e-12, 'limited');
        end

        function gain_shifts_the_pelvis_over_fixed_feet(testCase)
            % Moving the pelvis by a SHIFT and the legs by G*SHIFT
            % leaves each foot where it was, up to the damping.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            A = ws.getVariable('LegReferenceAngle');
            k = round(linspace(1, size(A, 2), 7));
            ref = struct('frames', k, 'q', A(:, k), 'pelvis_p', repmat([0.1; -0.2; 0.95], 1, numel(k)), ...
                'pelvis_R', repmat(local_rz(30), 1, 1, numel(k)));
            G = gs3dx_balance_gain(ref, ws);
            leak = zeros(3, numel(k), 2);
            for s = 1:2
                P = 'LR';
                geom = struct('mount_R', ws.getVariable([P(s) 'HipMountRotation']), ...
                    'mount_p', ws.getVariable([P(s) 'HipMountOffset']), ...
                    'thigh', ws.getVariable('ThighLength'), 'shank', ws.getVariable('ShankLength'));
                rows = (s - 1) * 6 + (1:6);
                for i = 1:numel(k)
                    for d = 1:3
                        shift = 0.01 * ((1:3).' == d);
                        [~, p0] = gs3dx_leg_fk(geom, ref.pelvis_R(:, :, i), ref.pelvis_p(:, i), ref.q(rows, i));
                        [~, p1] = gs3dx_leg_fk(geom, ref.pelvis_R(:, :, i), ref.pelvis_p(:, i) + shift, ...
                            ref.q(rows, i) + G(rows, :, i) * shift);
                        leak(d, i, s) = norm(p1 - p0) / 0.01;
                    end
                end
            end
            [worst, at] = max(leak(:));
            [d, i, s] = ind2sub(size(leak), at);
            fprintf('foot moves %.3f of the pelvis shift (median %.3f), worst axis %d frame %d leg %d\n', ...
                worst, median(leak, 'all'), d, k(i), s);
            % 2026-09-28: max 0.975 (vertical, lead leg, frame 545: the
            % knee is nearly straight, so a vertical shift cannot be
            % absorbed), median 0.303.
            testCase.verifyLessThan(max(leak, [], 'all'), 0.98, 'foot follows the pelvis');
            testCase.verifyLessThan(median(leak, 'all'), 0.31, 'foot follows the pelvis (median)');
        end

        function model_closes_the_balance_loop(testCase)
            m = testCase.mdl;
            ws = get_param(m, 'ModelWorkspace');
            blk = [m '/Lower Body/Leg Torque Commands'];
            testCase.verifyEqual(get_param(blk, 'SFBlockType'), 'MATLAB Function');
            chart = sfroot().find('-isa', 'Stateflow.EMChart', 'Path', blk);
            testCase.verifySubstring(chart.Script, 'gs3dx_balance_command');
            testCase.verifyEqual(get_param([m '/GS3DX Balance COM Goto'], 'GotoTag'), 'GS3DXBalanceCOM');
            testCase.verifyEqual(ws.getVariable('BalanceOn'), 1);
            testCase.verifyEqual([ws.getVariable('BalanceKp') ws.getVariable('BalanceKd')], [3 0.4]);
            n = numel(ws.getVariable('LegReferenceTime'));
            testCase.verifySize(ws.getVariable('BalanceCOMRef'), [3 n]);
            testCase.verifySize(ws.getVariable('BalanceGain'), [12 3 n]);
            testCase.verifySize(ws.getVariable('BalanceFootRef'), [6 n]);
            for P = 'LR'
                testCase.verifyEqual(get_param([m '/Lower Body/Balance ' P ' Ankle'], 'OutputSignals'), 'GlobalPosition');
            end
        end

        function balance_holds_the_centre_of_mass(testCase)
            % One simulation of the first 0.3 s.  The bounds pin the
            % 2026-09-28 result (docs/FIT.md).
            % GS3DX_CONTACT_CHECK closes the model, so read its workspace first.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            T = ws.getVariable('LegReferenceTime');
            Cref = ws.getVariable('BalanceCOMRef');
            start = ws.getVariable('TrackStart');
            c = gs3dx_contact_check(testCase.info, model=testCase.mdl, rest=true, stop_time=0.3, ...
                variables=start);
            e = vecnorm(c.com - interp1(T(:), Cref.', c.t(:)).');
            fprintf('0.3 s balance: COM error RMS %.1f max %.1f mm; slip %.1f/%.1f lift %.1f/%.1f mm; support %s\n', ...
                1000 * sqrt(mean(e .^ 2)), 1000 * max(e), ...
                1000 * [c.feet.L.slip c.feet.R.slip c.feet.L.lift c.feet.R.lift], mat2str(c.support, 3));
            testCase.verifyEqual(string(c.status), "success");
            testCase.verifyTrue(c.newton.pass, 'contacts and gravity are the only external forces');
            % 2026-09-28, Kp 3 / Kd 0.4: RMS 10.1 mm, max 16.5 mm (Kp 1: 14.6 / 24.9 mm).
            testCase.verifyLessThan(max(e), 0.0175, 'centre-of-mass error');
        end
    end
end

function [T, C0, Kp, Kd, A, R, Cref, Vref, Fref, G] = local_command_data()
T = [0 1];
C0 = (1:12).';
Kp = 100 * ones(12, 1);
Kd = 5 * ones(12, 1);
A = [zeros(12, 1) ones(12, 1)];
R = [ones(12, 1) 3 * ones(12, 1)];
Cref = [0 0.1; 0 0; 1 1];
Vref = [0.1 0.1; 0 0; 0 0];
Fref = [0.1 * ones(6, 1) 0.2 * ones(6, 1)];
G = repmat(reshape(1:24, 12, 2), 1, 1, 2);
end

function R = local_rz(deg)
c = cosd(deg);
s = sind(deg);
R = [c -s 0; s c 0; 0 0 1];
end
