classdef test_align_to_simulation_grid < matlab.unittest.TestCase
%TEST_ALIGN_TO_SIMULATION_GRID  Unit tests for the alignment helper.
%
%   Direct testing of private/align_to_simulation_grid.m via a small thunk
%   in the parent shared/ folder (private functions are accessible from
%   their parent folder's M-files).

    methods (TestClassSetup)
        function add_shared_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            shared_dir = fullfile(here, "..");
            addpath(shared_dir);
            testCase.addTeardown(@() rmpath(shared_dir));
        end
    end

    methods (Test)
        function test_synthetic_swing_aligns_at_specified_impact_time(testCase)
            % Build a synthetic swing whose clubhead speed peaks at t=0.40s.
            raw = synthesize_raw_swing(0.40, 1.0, 600);
            opts = default_align_options();
            opts.expected_impact_s = 0.25;
            aligned = call_align(raw, opts);

            t_at_impact = aligned.time(aligned.impact_idx);
            % Should land within 1 sample of expected_impact_s.
            tol = 1.5 / opts.sample_rate;
            testCase.verifyLessThan(abs(t_at_impact - 0.25), tol);
        end

        function test_resampling_preserves_starting_and_ending_pose(testCase)
            raw  = synthesize_raw_swing(0.40, 1.0, 600);
            opts = default_align_options();
            aligned = call_align(raw, opts);

            % Starting butt position should be near the raw butt at the
            % window-start frame (approximately raw.butt at t_impact - pre).
            t_start_raw = 0.40 - opts.pre_impact_s;
            t_start_raw = max(raw.time(1), t_start_raw);
            butt_expected = interp1(raw.time, raw.butt, t_start_raw, "linear");
            testCase.verifyLessThan(norm(aligned.butt(1, :) - butt_expected), 1e-3);

            t_end_raw = 0.40 + opts.post_impact_s;
            t_end_raw = min(raw.time(end), t_end_raw);
            butt_end_expected = interp1(raw.time, raw.butt, t_end_raw, "linear");
            testCase.verifyLessThan(norm(aligned.butt(end, :) - butt_end_expected), 1e-3);
        end

        function test_postconditions_hold(testCase)
            raw  = synthesize_raw_swing(0.40, 1.0, 600);
            opts = default_align_options();
            aligned = call_align(raw, opts);

            N = numel(aligned.time);
            testCase.verifyTrue(all(diff(aligned.time) > 0));
            testCase.verifyEqual(aligned.time(1), 0, "AbsTol", eps);
            testCase.verifyEqual(size(aligned.butt, 1), N);
            qn = sqrt(sum(aligned.club_quat.^2, 2));
            testCase.verifyLessThan(max(abs(qn - 1)), 1e-6);
        end
    end
end


function aligned = call_align(raw, opts)
    % Tunnel into private/ via a temporary helper-on-path technique:
    % copy align_to_simulation_grid out of private temporarily? Simpler —
    % invoke a public shim defined in shared/ that forwards to the private
    % function.  We avoid that by `cd`'ing into a folder whose parent is
    % shared/, so private/ is reachable.
    here = fileparts(mfilename("fullpath"));
    shared_dir = fullfile(here, "..");
    old = pwd;
    c = onCleanup(@() cd(old));
    cd(shared_dir);
    aligned = align_to_simulation_grid(raw, opts);
end


function raw = synthesize_raw_swing(impact_t, total_t, n_frames)
    % Simple model: clubhead moves on an arc whose angular speed peaks at
    % impact_t, butt traces a smaller arc.  Quaternion taken from the
    % shaft direction.
    t = linspace(0, total_t, n_frames).';
    omega = 8 * exp(-((t - impact_t) / 0.10).^2);
    theta = cumtrapz(t, omega);

    R_butt  = 0.8;   % shoulder-to-grip radius (m)
    R_head  = 1.5;
    butt = [R_butt * cos(theta), R_butt * sin(theta), zeros(numel(t), 1)];
    head = [R_head * cos(theta), R_head * sin(theta), zeros(numel(t), 1)];

    % Quaternion from shaft vector
    quats = zeros(numel(t), 4);
    for k = 1:numel(t)
        z = head(k, :) - butt(k, :);
        z = z / norm(z);
        x = [0, 0, 1];
        x = x - dot(x, z) * z;
        x = x / norm(x);
        y = cross(z, x);
        Rk = [x.', y.', z.'];
        tr = Rk(1,1) + Rk(2,2) + Rk(3,3);
        if tr > 0
            S = 2 * sqrt(tr + 1.0);
            quats(k, :) = [0.25*S, (Rk(3,2)-Rk(2,3))/S, (Rk(1,3)-Rk(3,1))/S, (Rk(2,1)-Rk(1,2))/S];
        else
            quats(k, :) = [1, 0, 0, 0];
        end
        if quats(k, 1) < 0
            quats(k, :) = -quats(k, :);
        end
    end

    raw = struct("time", t, "butt", butt, "clubhead", head, "club_quat", quats);
end
