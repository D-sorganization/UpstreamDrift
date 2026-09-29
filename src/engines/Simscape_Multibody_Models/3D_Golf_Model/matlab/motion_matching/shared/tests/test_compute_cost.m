classdef test_compute_cost < matlab.unittest.TestCase
%TEST_COMPUTE_COST  Unit tests for compute_cost (see COST_FUNCTION_SPEC.md).

    methods (Test)
        function test_zero_difference_yields_zero_cost(testCase)
            % Sim output identical to target -> all kinematic terms = 0,
            % only lambda * regularizer remains.
            tgt = make_target();
            sim = sim_matches_target(tgt);
            opts = default_cost_options();
            opts.lambda = 0;  % isolate kinematic terms
            [J, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);
            testCase.verifyEqual(J, 0, "AbsTol", 1e-12);
            testCase.verifyEqual(terms.position, 0, "AbsTol", 1e-12);
            testCase.verifyEqual(terms.orientation, 0, "AbsTol", 1e-12);
            testCase.verifyEqual(terms.impact_anchor, 0, "AbsTol", 1e-12);
        end

        function test_position_term_is_squared_metres(testCase)
            % Single-frame target+sim. 1 mm offset on butt only on 1 frame.
            % Mean over N frames of ||db||^2 + ||dc||^2.
            % With only butt offset (1e-3 m, x-axis) on frame 1 of N=5:
            % position term mean = (1e-6 + 0 + 0 + 0 + 0) / 5 = 2e-7 m^2.
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.butt(1, 1) = sim.butt(1, 1) + 1e-3;
            opts = default_cost_options();
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 0;
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);
            N = numel(tgt.time);
            expected = opts.w_position * (1e-6 / N);
            testCase.verifyEqual(terms.position, expected, "AbsTol", 1e-15);
        end

        function test_orientation_term_uses_geodesic_distance(testCase)
            % q vs -q is the same rotation -> angle = 0, term = 0.
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.club_quat = -sim.club_quat;  % flip every quaternion sign
            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 0;
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);
            testCase.verifyEqual(terms.orientation, 0, "AbsTol", 1e-12);
        end

        function test_impact_anchor_amplifies_impact_frame_error(testCase)
            % Put a 1 cm grip error only at the impact frame.  The
            % anchor term now follows the grip (the rigid body→club
            % interface) rather than the clubhead, since clubhead
            % residuals can come from club-length / shaft-flex
            % differences that we do not want the cost to chase.
            tgt = make_target();
            sim = sim_matches_target(tgt);
            k = tgt.impact_idx;
            sim.butt(k, 1) = sim.butt(k, 1) + 0.01;     % butt is alias of grip
            opts = default_cost_options();
            opts.w_position           = 0;
            opts.w_position_grip      = 0;
            opts.w_position_clubhead  = 0;
            opts.w_orientation        = 0;
            opts.w_orientation_grip   = 0;
            opts.w_orientation_club   = 0;
            opts.lambda               = 0;
            [~, terms] = compute_cost(zeros(7, 1), tgt, @(~) sim, opts);
            % anchor = w_anchor_impact * ||d||^2 = 10 * (0.01)^2 = 1e-3
            testCase.verifyEqual(terms.impact_anchor, 1e-3, "AbsTol", 1e-12);
        end

        function test_regularizer_total_work_reduces_with_smaller_torques(testCase)
            tgt = make_target();
            opts = default_cost_options();
            opts.w_position = 0;
            opts.w_orientation = 0;
            opts.w_anchor_impact = 0;
            opts.lambda = 1.0;  % isolate regularizer
            big   = sim_matches_target(tgt);
            small = sim_matches_target(tgt);
            big.tau   = big.tau * 10;
            [~, t_big]   = compute_cost(zeros(7, 1), tgt, @(~) big, opts);
            [~, t_small] = compute_cost(zeros(7, 1), tgt, @(~) small, opts);
            testCase.verifyGreaterThan(t_big.regularizer, t_small.regularizer);
        end

        function test_regularizer_choice_switch(testCase)
            % Each regularizer mode must produce a distinct value on a
            % non-trivial input.
            tgt = make_target();
            sim = sim_matches_target(tgt);
            theta = linspace(0.1, 0.7, 7).';
            modes = ["total_work", "peak_power", "torque_l2", "coeff_l2"];
            vals = zeros(1, numel(modes));
            for i = 1:numel(modes)
                opts = default_cost_options();
                opts.w_position = 0;
                opts.w_orientation = 0;
                opts.w_anchor_impact = 0;
                opts.lambda = 1.0;
                opts.regularizer = modes(i);
                [~, terms] = compute_cost(theta, tgt, @(~) sim, opts);
                vals(i) = terms.regularizer;
            end
            testCase.verifyEqual(numel(unique(vals)), numel(modes), ...
                "All regularizer modes should yield distinct values.");
            testCase.verifyTrue(all(vals >= 0));
        end

        function test_terms_struct_breakdown_sums_to_total(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.butt = sim.butt + 0.005;
            sim.clubhead = sim.clubhead + 0.003;
            theta = 0.1 * ones(7, 1);
            [J, terms] = compute_cost(theta, tgt, @(~) sim);
            s = terms.position + terms.orientation + ...
                terms.impact_anchor + terms.regularizer;
            testCase.verifyEqual(terms.total, J, "AbsTol", eps);
            testCase.verifyEqual(s, J, "RelTol", 1e-12, "AbsTol", 1e-15);
        end

        function test_invalid_target_struct_rejected(testCase)
            bad = struct("butt", zeros(5, 3));  % missing nearly everything
            testCase.verifyError( ...
                @() compute_cost(zeros(7, 1), bad, @(t) struct()), ...
                "validator:missingField");
        end

        function test_nan_theta_rejected(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            theta = [1; NaN; 3; 4; 5; 6; 7];
            testCase.verifyError( ...
                @() compute_cost(theta, tgt, @(~) sim), ...
                "validator:notFinite");
        end

        function test_J_is_finite_nonneg(testCase)
            tgt = make_target();
            sim = sim_matches_target(tgt);
            sim.butt = sim.butt + randn(size(sim.butt)) * 1e-3;
            J = compute_cost(0.05 * ones(7, 1), tgt, @(~) sim);
            testCase.verifyTrue(isfinite(J));
            testCase.verifyGreaterThanOrEqual(J, 0);
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
