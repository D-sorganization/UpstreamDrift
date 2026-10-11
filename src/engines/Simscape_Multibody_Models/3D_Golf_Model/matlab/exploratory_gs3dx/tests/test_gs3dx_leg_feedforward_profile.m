classdef test_gs3dx_leg_feedforward_profile < matlab.unittest.TestCase
%TEST_GS3DX_LEG_FEEDFORWARD_PROFILE Unit tests for time-varying leg feedforward command.
% Tests backward compatibility with constant 12-vectors (column and row),
% time-varying profiles (endpoints, interior linear interpolation, extrapolation clamping),
% zero initial ramp, balance-off retention, and contract failure on invalid shape or nonfinite values.

    methods (TestClassSetup)
        function setupPath(testCase)
            original_path=path;
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
            testCase.addTeardown(@() path(original_path));
        end
    end

    methods (Test)
        function test_constant_column_and_row_equivalence(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            c_col = (1:12).';
            c_row = 1:12;
            times = [0, 0.25, 0.5, 0.8, 1.0];
            for t = times
                [cmd_col, s_col] = gs3dx_balance_command(t, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                    T, c_col, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 1);
                [cmd_row, s_row] = gs3dx_balance_command(t, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                    T, c_row, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 1);
                testCase.verifyEqual(cmd_row, cmd_col, 'AbsTol', 1e-12);
                testCase.verifyEqual(s_row, s_col, 'AbsTol', 1e-12);
            end
        end

        function test_time_varying_profile_endpoints_and_interior_interp(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            % T = [0, 0.5, 1.0] (3 frames)
            v1 = (1:12).';
            v2 = (13:24).';
            v3 = (25:36).';
            C_prof = [v1, v2, v3];

            % At t = 0 (frame 1)
            [cmd_0, ~] = gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_base_0, ~] = gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v1, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_0, cmd_base_0, 'AbsTol', 1e-12);

            % At t = 0.25 (midpoint between frame 1 and 2)
            v_mid1 = 0.5 * v1 + 0.5 * v2;
            [cmd_025, ~] = gs3dx_balance_command(0.25, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_base_025, ~] = gs3dx_balance_command(0.25, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v_mid1, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_025, cmd_base_025, 'AbsTol', 1e-12);

            % At t = 0.5 (frame 2)
            [cmd_05, ~] = gs3dx_balance_command(0.5, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_base_05, ~] = gs3dx_balance_command(0.5, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v2, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_05, cmd_base_05, 'AbsTol', 1e-12);

            % At t = 1.0 (frame 3)
            [cmd_1, ~] = gs3dx_balance_command(1.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_base_1, ~] = gs3dx_balance_command(1.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v3, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_1, cmd_base_1, 'AbsTol', 1e-12);
        end

        function test_clamping_outside_time_grid(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            v1 = (1:12).';
            v2 = (13:24).';
            v3 = (25:36).';
            C_prof = [v1, v2, v3];

            % Clamped below T(1)
            [cmd_pre, ~] = gs3dx_balance_command(-0.5, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_t0, ~] = gs3dx_balance_command(0.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v1, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_pre, cmd_t0, 'AbsTol', 1e-12);

            % Clamped above T(end)
            [cmd_post, ~] = gs3dx_balance_command(2.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_tend, ~] = gs3dx_balance_command(1.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, v3, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_post, cmd_tend, 'AbsTol', 1e-12);
        end

        function test_zero_initial_ramp(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            % Ramped from 0 at t=0 to target torque at t=1.0
            target_tau = 50 * ones(12, 1);
            C_ramp = [zeros(12, 1), 0.5 * target_tau, target_tau];

            [cmd_0, ~] = gs3dx_balance_command(0.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_ramp, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            [cmd_zero, ~] = gs3dx_balance_command(0.0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, zeros(12, 1), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_0, cmd_zero, 'AbsTol', 1e-12, ...
                'At t=0, feedforward with zero start must add zero torque.');

            [cmd_half, ~] = gs3dx_balance_command(0.5, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, C_ramp, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            cmd_zero_half=gs3dx_balance_command(0.5, Cref(:,1), Vref(:,1), Fref(:,1), ...
                T, zeros(12,1), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0);
            testCase.verifyEqual(cmd_half - cmd_zero_half, 0.5 * target_tau, 'AbsTol', 1e-12);
        end

        function test_offbalance_feedforward_retained(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            c_prof = repmat((1:12).', 1, 3);
            [cmd, shift] = gs3dx_balance_command(0.5, [0.3; -0.2; 1], [1; 1; 0], ones(6, 1), ...
                T, c_prof, Kp, Kd, A, R, Cref, Vref, Fref, G, 1, 0.2, 1, 0.1, 0);
            testCase.verifyEqual(shift, [0; 0], 'AbsTol', 1e-12);
            % Balance is off, but feedforward torque is retained
            expected_cmd = c_prof(:, 2) + Kp .* A(:, 2) + Kd .* R(:, 2);
            testCase.verifyEqual(cmd, expected_cmd, 'AbsTol', 1e-12);
        end

        function test_malformed_shape_fails(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            % Wrong number of rows/elements
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, zeros(11, 1), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:invalidShape');
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, zeros(13, 1), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:invalidShape');
            % Wrong column count for profile (T has 3 columns, C has 4)
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, zeros(12, 4), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:invalidShape');
            % Arbitrary matrix
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, zeros(2, 6), Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:invalidShape');
            % Empty
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, [], Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:invalidShape');
        end

        function test_nonfinite_fails(testCase)
            [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs();
            c_nan = (1:12).';
            c_nan(5) = NaN;
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, c_nan, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:nonfinite');

            c_inf = zeros(12, 3);
            c_inf(2, 2) = Inf;
            testCase.verifyError(@() gs3dx_balance_command(0, Cref(:, 1), Vref(:, 1), Fref(:, 1), ...
                T, c_inf, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit, 0), ...
                'gs3dx:balance_command:nonfinite');
        end

        function test_complex_type_fails(testCase)
            [T,Kp,Kd,A,R,Cref,Vref,Fref,G,kp,kd,kf,limit]=local_base_inputs();
            testCase.verifyError(@() gs3dx_balance_command(0,Cref(:,1),Vref(:,1),Fref(:,1), ...
                T,1i*ones(12,1),Kp,Kd,A,R,Cref,Vref,Fref,G,kp,kd,kf,limit,0), ...
                'gs3dx:balance_command:invalidType');
        end

        function test_higher_dimension_shape_fails(testCase)
            [T,Kp,Kd,A,R,Cref,Vref,Fref,G,kp,kd,kf,limit]=local_base_inputs();
            testCase.verifyError(@() gs3dx_balance_command(0,Cref(:,1),Vref(:,1),Fref(:,1), ...
                T,zeros(12,3,2),Kp,Kd,A,R,Cref,Vref,Fref,G,kp,kd,kf,limit,0), ...
                'gs3dx:balance_command:invalidShape');
        end
    end
end

function [T, Kp, Kd, A, R, Cref, Vref, Fref, G, kp, kd, kf, limit] = local_base_inputs()
T = [0, 0.5, 1.0];
Kp = 100 * ones(12, 1);
Kd = 5 * ones(12, 1);
A = [zeros(12, 1), ones(12, 1), 2 * ones(12, 1)];
R = [ones(12, 1), 2 * ones(12, 1), 3 * ones(12, 1)];
Cref = [0 0.05 0.1; 0 0 0; 1 1 1];
Vref = [0.1 0.1 0.1; 0 0 0; 0 0 0];
Fref = repmat([0.1 * ones(3, 1); -0.1 * ones(3, 1)], 1, 3);
G = repmat(reshape(1:24, 12, 2), 1, 1, 3);
kp = 1;
kd = 0.2;
kf = 1;
limit = 0.1;
end
