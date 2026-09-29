classdef test_compute_total_work < matlab.unittest.TestCase
%TEST_COMPUTE_TOTAL_WORK  Unit tests for compute_total_work.

    methods (Test)
        function test_static_pose_yields_zero_work(testCase)
            % Zero torques and zero velocities -> zero work.
            sim = make_sim_out(linspace(0, 0.3, 31), zeros(31, 3), zeros(31, 3));
            W = compute_total_work(sim);
            testCase.verifyEqual(W, 0, "AbsTol", 1e-12);
        end

        function test_constant_torque_constant_velocity_known_value(testCase)
            % tau = 10 N*m, omega = 2 rad/s, dt = 0.3 s, single joint.
            % power = |10*2| = 20 W. work = 20 * 0.3 = 6 J.
            t = linspace(0, 0.3, 31).';
            tau = 10 * ones(numel(t), 1);
            om  =  2 * ones(numel(t), 1);
            sim = make_sim_out(t, tau, om);
            W = compute_total_work(sim);
            testCase.verifyEqual(W, 6.0, "AbsTol", 1e-9);
        end

        function test_negative_work_counted_positively(testCase)
            % Eccentric work: tau and omega opposite sign should still add.
            t = linspace(0, 1, 101).';
            tau = -3 * ones(numel(t), 1);
            om  =  4 * ones(numel(t), 1);
            sim = make_sim_out(t, tau, om);
            W = compute_total_work(sim);
            % |tau*omega| = 12, integrated over 1 s -> 12 J.
            testCase.verifyEqual(W, 12.0, "AbsTol", 1e-9);
        end

        function test_multiple_joints_sum_correctly(testCase)
            t = linspace(0, 1, 11).';
            tau = [ones(11, 1) * 5, ones(11, 1) * -2];
            om  = [ones(11, 1) * 1, ones(11, 1) *  3];
            % Per-frame: |5*1| + |-2*3| = 5 + 6 = 11; * 1 s = 11 J.
            sim = make_sim_out(t, tau, om);
            W = compute_total_work(sim);
            testCase.verifyEqual(W, 11.0, "AbsTol", 1e-9);
        end

        function test_missing_field_errors(testCase)
            bad = struct("time", linspace(0, 1, 5).', "tau", zeros(5, 1));
            testCase.verifyError(@() compute_total_work(bad), ...
                "validator:missingField");
        end
    end
end

function sim = make_sim_out(t, tau, om)
    sim = struct();
    sim.time  = t(:);
    sim.tau   = tau;
    sim.omega = om;
end
