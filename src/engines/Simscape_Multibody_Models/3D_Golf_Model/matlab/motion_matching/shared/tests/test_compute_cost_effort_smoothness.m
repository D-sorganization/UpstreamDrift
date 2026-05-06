classdef test_compute_cost_effort_smoothness < matlab.unittest.TestCase
%TEST_COMPUTE_COST_EFFORT_SMOOTHNESS  Unit tests for the effort_l2 and
%   smoothness_l2 regularizers added for parity with PR #3966 in
%   MachineLearning/optimize_torque_sequence_for_club.py.

    methods (Test)
        function test_effort_l2_zero_reference_matches_torque_l2_mean(testCase)
            % effort_l2 with tau_reference=[] (zero) and unit weights equals
            % mean(tau .^ 2, 'all').
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.tau = reshape(linspace(-1, 2, numel(sim.tau)), size(sim.tau));

            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1;
            opts.regularizer = "effort_l2";
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);

            expected = mean(sim.tau .^ 2, 'all');
            testCase.verifyEqual(terms.regularizer, expected, "AbsTol", 1e-12);
        end

        function test_smoothness_l2_zero_for_constant_torque(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.tau = 0.7 * ones(size(sim.tau));

            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1;
            opts.regularizer = "smoothness_l2";
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);

            testCase.verifyEqual(terms.regularizer, 0, "AbsTol", 1e-15);
        end

        function test_smoothness_l2_ramp_analytic(testCase)
            % tau(t_i) = a * t_i with t = linspace(0, 0.3, N) on a single
            % joint. diff(tau) = a*dt for every step, so
            % mean(diff(tau).^2, 'all') = (a*dt)^2.
            tgt = make_target();
            N = numel(tgt.time);
            sim = sim_matches_target(tgt);
            a = 4.5;
            dt = tgt.time(2) - tgt.time(1);
            sim.tau = a * tgt.time(:);   % N x 1

            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1;
            opts.regularizer = "smoothness_l2";
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);

            expected = (a * dt) ^ 2;
            testCase.verifyEqual(terms.regularizer, expected, ...
                                 "AbsTol", 1e-12, "RelTol", 1e-12);
            testCase.verifyEqual(N - 1, size(sim.tau, 1) - 1);  % sanity
        end

        function test_regularizer_weights_honoured(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            % 3 joints; populate tau with distinct constants per joint.
            N = numel(tgt.time);
            sim.tau = [ones(N, 1), 2 * ones(N, 1), 3 * ones(N, 1)];

            % effort_l2 with zero reference; weighted mean of tau^2.
            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1;
            opts.regularizer = "effort_l2";
            opts.regularizer_weights = [0.5, 1.0, 2.0];
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);

            expected = mean(sim.tau .^ 2 .* [0.5, 1.0, 2.0], 'all');
            testCase.verifyEqual(terms.regularizer, expected, "AbsTol", 1e-12);
        end

        function test_effort_l2_with_nonzero_reference(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.tau = ones(size(sim.tau));
            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1;
            opts.regularizer = "effort_l2";
            opts.tau_reference = ones(size(sim.tau));
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);
            testCase.verifyEqual(terms.regularizer, 0, "AbsTol", 1e-15);
        end
    end
end

% ----- helpers -----

function tgt = make_target()
    N = 31;
    tgt = struct();
    tgt.time     = linspace(0, 0.3, N).';
    tgt.butt     = repmat([0.0, 0.0, 1.2], N, 1) + 0.001 * (1:N).' * [1 0 0];
    tgt.clubhead = repmat([0.0, 0.0, 0.2], N, 1) + 0.002 * (1:N).' * [0 1 0];
    tgt.club_quat = repmat([1, 0, 0, 0], N, 1);
    tgt.impact_idx = uint32(round(N * 0.8));
end

function sim = sim_matches_target(tgt)
    N = numel(tgt.time);
    sim = struct();
    sim.time      = tgt.time;
    sim.butt      = tgt.butt;
    sim.clubhead  = tgt.clubhead;
    sim.club_quat = tgt.club_quat;
    sim.tau       = ones(N, 3);
    sim.omega     = ones(N, 3) * 0.5;
end
