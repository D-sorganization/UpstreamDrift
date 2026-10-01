classdef test_gs3dx_xyz_map < matlab.unittest.TestCase
%TEST_GS3DX_XYZ_MAP  Spherical <-> Gimbal (intrinsic X-Y-Z) kinematics (#10955).
%
%   Pure math, no Simulink: the rotation, rate, acceleration and torque maps
%   are checked against rotation matrices built directly from the angles and
%   against finite differences.

    properties (Constant)
        % [a b c] trajectories in rad: a(t) = a0 + a1 t + a2 t^2 per angle.
        A0 = [0.7, -0.4, 2.9]
        A1 = [1.3, 0.8, -2.1]
        A2 = [-0.9, 0.5, 1.7]
    end

    methods (TestClassSetup)
        function setup(~)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
        end
    end

    methods (Test)
        function rate_matrix_is_identity_at_zero(testCase)
            testCase.verifyEqual(gs3dx_xyz_rate_matrix(0, 0), eye(3), 'AbsTol', 1e-15);
        end

        function angles_are_recovered_from_quaternion(testCase)
            ang = testCase.A0;
            [~, out] = gs3dx_xyz_map(local_quat(ang), zeros(3, 1), zeros(3, 1), ...
                zeros(3, 1), zeros(3, 1), rad2deg(ang));
            testCase.verifyEqual(out.', rad2deg(ang), 'AbsTol', 1e-10);
        end

        function first_and_third_angles_follow_reference_branch(testCase)
            ang = testCase.A0;
            ref = rad2deg(ang) + [-360, 0, 720] + [40, 0, -30];
            [~, out] = gs3dx_xyz_map(local_quat(ang), zeros(3, 1), zeros(3, 1), ...
                zeros(3, 1), zeros(3, 1), ref);
            testCase.verifyEqual(out.', rad2deg(ang) + [-360, 0, 720], 'AbsTol', 1e-10);
        end

        function rates_invert_finite_difference_angular_velocity(testCase)
            t = 0.37;
            [ang, rate] = testCase.trajectory(t);
            w = local_omega(@(s) testCase.trajectory(s), t);
            [~, ~, qd] = gs3dx_xyz_map(local_quat(ang), w, zeros(3, 1), ...
                zeros(3, 1), zeros(3, 1), rad2deg(ang));
            testCase.verifyEqual(qd.', rad2deg(rate), 'RelTol', 1e-7);
        end

        function accelerations_invert_finite_difference(testCase)
            t = 0.37; h = 1e-5;
            [ang, ~, acc] = testCase.trajectory(t);
            traj = @(s) testCase.trajectory(s);
            b = (local_omega(traj, t + h) - local_omega(traj, t - h)) / (2 * h);
            w = local_omega(traj, t);
            [~, ~, ~, qdd] = gs3dx_xyz_map(local_quat(ang), w, b, ...
                zeros(3, 1), zeros(3, 1), rad2deg(ang));
            testCase.verifyEqual(qdd.', rad2deg(acc), 'RelTol', 1e-5);
        end

        function torque_preserves_power(testCase)
            % Same power as the Gimbal: T . omega_F == tau . dq/dt.
            t = 0.37;
            [ang, rate] = testCase.trajectory(t);
            w = gs3dx_xyz_rate_matrix(ang(2), ang(3)) * rate.';
            tau = [12; -7; 3];
            T = gs3dx_xyz_map(local_quat(ang), w, zeros(3, 1), tau, zeros(3, 1), rad2deg(ang));
            testCase.verifyEqual(T.' * w, tau.' * rate.', 'RelTol', 1e-12);
            % Generalized force on each Euler axis equals tau.
            testCase.verifyEqual(gs3dx_xyz_rate_matrix(ang(2), ang(3)).' * T, tau, 'RelTol', 1e-12);
        end

        function damping_opposes_rates_in_degrees(testCase)
            t = 0.37;
            [ang, rate] = testCase.trajectory(t);
            w = gs3dx_xyz_rate_matrix(ang(2), ang(3)) * rate.';
            c = [0.5; 2; 10];
            T = gs3dx_xyz_map(local_quat(ang), w, zeros(3, 1), zeros(3, 1), c, rad2deg(ang));
            E = gs3dx_xyz_rate_matrix(ang(2), ang(3));
            testCase.verifyEqual(E.' * T, -c .* rad2deg(rate.'), 'RelTol', 1e-12);
        end
    end

    methods
        function [ang, rate, acc] = trajectory(testCase, t)
            ang  = testCase.A0 + testCase.A1 * t + testCase.A2 * t^2;
            rate = testCase.A1 + 2 * testCase.A2 * t;
            acc  = 2 * testCase.A2;
        end
    end
end

function R = local_rotm(ang)
    a = ang(1); b = ang(2); c = ang(3);
    Rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
    Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)];
    Rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1];
    R = Rx * Ry * Rz;
end

function q = local_quat(ang)
% Unit quaternion [w x y z] of R(ang) (Shepperd's method, no toolbox).
    R = local_rotm(ang);
    [~, k] = max([trace(R), R(1, 1), R(2, 2), R(3, 3)]);
    switch k
        case 1
            s = 2 * sqrt(1 + trace(R));
            q = [s / 4; (R(3, 2) - R(2, 3)) / s; (R(1, 3) - R(3, 1)) / s; (R(2, 1) - R(1, 2)) / s];
        case 2
            s = 2 * sqrt(1 + R(1, 1) - R(2, 2) - R(3, 3));
            q = [(R(3, 2) - R(2, 3)) / s; s / 4; (R(1, 2) + R(2, 1)) / s; (R(1, 3) + R(3, 1)) / s];
        case 3
            s = 2 * sqrt(1 + R(2, 2) - R(1, 1) - R(3, 3));
            q = [(R(1, 3) - R(3, 1)) / s; (R(1, 2) + R(2, 1)) / s; s / 4; (R(2, 3) + R(3, 2)) / s];
        otherwise
            s = 2 * sqrt(1 + R(3, 3) - R(1, 1) - R(2, 2));
            q = [(R(2, 1) - R(1, 2)) / s; (R(1, 3) + R(3, 1)) / s; (R(2, 3) + R(3, 2)) / s; s / 4];
    end
end

function w = local_omega(traj, t)
% Follower-frame angular velocity from central differences of R(t).
    h = 1e-6;
    R = local_rotm(traj(t));
    Rdot = (local_rotm(traj(t + h)) - local_rotm(traj(t - h))) / (2 * h);
    W = R.' * Rdot;
    w = [W(3, 2); W(1, 3); W(2, 1)];
end
