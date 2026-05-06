classdef test_plot_fit_quality_card < matlab.unittest.TestCase
%TEST_PLOT_FIT_QUALITY_CARD  Headless tests for View 3 summary card (#3991).
%
%   Tagged 'slow' tests do disk I/O (PNG/FIG export); the rest are fast.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared = fileparts(here);
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function test_returns_single_figure_handle(testCase)
            [res, tgt] = make_pair();
            fig = plot_fit_quality_card(res, tgt, headless_opts());
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            testCase.verifyClass(fig, 'matlab.ui.Figure');
            testCase.verifyTrue(isgraphics(fig, 'figure'));
        end

        function test_card_includes_all_required_metrics_text(testCase)
            [res, tgt] = make_pair();
            fig = plot_fit_quality_card(res, tgt, headless_opts());
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            txt = local_collect_card_text(fig);
            testCase.verifySubstring(txt, "RMSE");
            testCase.verifySubstring(txt, "clubhead");
            testCase.verifySubstring(txt, "butt");
            testCase.verifySubstring(txt, "orientation");
            testCase.verifySubstring(txt, "speed");
            testCase.verifySubstring(txt, "Total work");
            testCase.verifySubstring(txt, "Peak joint power");
            testCase.verifySubstring(txt, "Solver");
            testCase.verifySubstring(txt, "Iterations");
            testCase.verifySubstring(txt, "Wall clock");
        end

        function test_thumbnail_axes_present(testCase)
            [res, tgt] = make_pair();
            fig = plot_fit_quality_card(res, tgt, headless_opts());
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            v1 = findall(fig, 'Type', 'axes', 'Tag', 'card_thumb_view1');
            v2 = findall(fig, 'Type', 'axes', 'Tag', 'card_thumb_view2');
            testCase.verifyGreaterThanOrEqual(numel(v1), 1, ...
                'View 1 thumbnail axes missing');
            testCase.verifyGreaterThanOrEqual(numel(v2), 1, ...
                'View 2 thumbnail axes missing');
        end

        function test_card_includes_short_provenance_in_footer(testCase)
            [res, tgt] = make_pair();
            res.target_hash = "abcdef0123456789aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
            res.git_commit  = "deadbeefcafebabe0000";
            res.branch      = "feat/motion-matching/022";
            fig = plot_fit_quality_card(res, tgt, headless_opts());
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            footer = findall(fig, 'Tag', 'card_footer');
            testCase.assertNotEmpty(footer);
            txt = strjoin(string(footer(1).String), newline);
            testCase.verifySubstring(txt, "abcdef0");
            testCase.verifySubstring(txt, "deadbee");
            testCase.verifySubstring(txt, "feat/motion-matching/022");
        end

        function test_save_path_writes_png_and_fig(testCase)
            [res, tgt] = make_pair();
            opts = headless_opts();
            opts.save_to_disk = true;
            opts.output_dir = string(tempname);
            mkdir(opts.output_dir);
            testCase.addTeardown(@() local_rmdir(opts.output_dir));
            res.swing_id = "unit_test_swing";

            fig = plot_fit_quality_card(res, tgt, opts);
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>

            png_path = fullfile(opts.output_dir, "unit_test_swing.png");
            fig_path = fullfile(opts.output_dir, "unit_test_swing.fig");
            testCase.verifyTrue(isfile(png_path), ...
                sprintf("expected PNG at %s", png_path));
            testCase.verifyTrue(isfile(fig_path), ...
                sprintf("expected FIG at %s", fig_path));
        end

        function test_handles_missing_optional_result_fields_gracefully(testCase)
            [res, tgt] = make_pair();
            % Strip every optional field; keep only the required provenance.
            res = rmfield_safe(res, ["peak_joint_power_W", "peak_joint_name", ...
                "iterations", "swing_id", "branch", "final_total_work_J"]);
            fig = plot_fit_quality_card(res, tgt, headless_opts());
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            testCase.verifyTrue(isgraphics(fig, 'figure'));
            txt = local_collect_card_text(fig);
            % "n/a" must appear in place of missing peak power.
            testCase.verifySubstring(txt, "n/a");
        end

        function test_rejects_result_missing_provenance(testCase)
            [res, tgt] = make_pair();
            bad = rmfield(res, 'git_commit');
            testCase.verifyError( ...
                @() plot_fit_quality_card(bad, tgt, headless_opts()), ...
                'validator:missingField');
        end

        function test_runs_headless_visible_off(testCase)
            [res, tgt] = make_pair();
            opts = headless_opts();
            fig = plot_fit_quality_card(res, tgt, opts);
            cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
            testCase.verifyEqual(get(fig, 'Visible'), 'off');
        end
    end
end

% ===================== helpers =====================

function [res, tgt] = make_pair()
    tgt = make_target();
    N = numel(tgt.time);
    sim = struct();
    sim.time       = tgt.time;
    sim.r_butt     = tgt.butt + 1e-3 * randn(N, 3);
    sim.r_clubhead = tgt.clubhead + 1e-3 * randn(N, 3);
    sim.q_club     = tgt.club_quat;
    sim.tau        = 0.5 * ones(N, 3);
    sim.omega      = 0.5 * ones(N, 3);
    res = struct();
    res.sim_out               = sim;
    res.solver                = "fmincon-sqp";
    res.solver_options        = struct('iterations', 247);
    res.iterations            = 247;
    res.target_hash           = "7a3f1b2c8d9e4f5061728394a5b6c7d8e9f0112233445566778899aabbccddee";
    res.git_commit            = "abcdef1234567890";
    res.matlab_version        = "9.13";
    res.duration_s            = 252;
    res.timestamp_utc         = "2026-05-05T12:00:00Z";
    res.coefficients          = zeros(3, 7);
    res.final_rmse_m          = 0.0023;
    res.final_total_work_J    = 284;
    res.peak_joint_power_W    = 1200;
    res.peak_joint_name       = "LE";
    res.swing_id              = "TW_ProV1";
    res.branch                = "feat/motion-matching/022-fit-quality-card";
end

function tgt = make_target()
    N = 31;
    tgt = struct();
    tgt.time       = linspace(0, 0.3, N).';
    tgt.butt       = repmat([0.0, 0.0, 1.2], N, 1) + 0.001 * (1:N).' * [1 0 0];
    tgt.clubhead   = repmat([0.0, 0.0, 0.2], N, 1) + 0.002 * (1:N).' * [0 1 0];
    tgt.club_quat  = repmat([1, 0, 0, 0], N, 1);
    tgt.impact_idx = uint32(round(N * 0.8));
end

function opts = headless_opts()
    opts = default_viz_options();
    opts.visible = "off";
    opts.figure_visible = "off";
end

function txt = local_collect_card_text(fig)
    tags = ["card_header", "card_metrics", "card_regularizer", "card_footer"];
    parts = strings(0, 1);
    for tg = tags
        h = findall(fig, 'Tag', char(tg));
        for k = 1:numel(h)
            try
                s = string(h(k).String);
                parts = [parts; s(:)]; %#ok<AGROW>
            catch
                % skip handles without a String property
            end
        end
    end
    txt = strjoin(parts, newline);
end

function s = rmfield_safe(s, names)
    for n = string(names)
        if isfield(s, n)
            s = rmfield(s, n);
        end
    end
end

function local_rmdir(d)
    try
        if isfolder(d)
            rmdir(d, 's');
        end
    catch
        % best-effort cleanup
    end
end
