classdef test_animate_trajectory_overlay < matlab.unittest.TestCase
%TEST_ANIMATE_TRAJECTORY_OVERLAY  Headless tests for View 1 animated.
%
%   The MP4 save test is tagged Slow so it is excluded from the fast
%   suite; CI selects it explicitly when running the full suite.
%
%   GitHub issue: #3989.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared = fileparts(here);
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function test_animate_interactive_returns_figure(testCase)
            [result, target, opts] = make_inputs();
            opts.save_path = "";  % interactive mode
            out = animate_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close_if_valid(out));
            testCase.verifyTrue(isgraphics(out, 'figure'));
        end

        function test_animate_does_not_modify_inputs(testCase)
            [result, target, opts] = make_inputs();
            r0 = result;
            t0 = target;
            opts.save_path = "";
            out = animate_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close_if_valid(out));
            testCase.verifyEqual(result, r0);
            testCase.verifyEqual(target, t0);
        end
    end

    methods (Test, TestTags = {'Slow'})
        function test_animate_save_writes_mp4(testCase)
            [result, target, opts] = make_inputs();
            tmp = [tempname() '.mp4'];
            opts.save_path = tmp;
            opts.video_fps = 15;
            % Some CI runners lack an MPEG-4 codec; fall back to
            % Motion JPEG AVI in that case so the test still exercises
            % the writer-handle return contract.
            try
                vw = animate_trajectory_overlay(result, target, opts);
            catch err
                if contains(err.message, 'MPEG-4', 'IgnoreCase', true)
                    opts.video_format = "Motion JPEG AVI";
                    opts.save_path = [tempname() '.avi'];
                    tmp = char(opts.save_path);
                    vw = animate_trajectory_overlay(result, target, opts);
                else
                    rethrow(err);
                end
            end
            cleanup = onCleanup(@() safe_close_writer(vw));
            testCase.verifyClass(vw, 'VideoWriter');
            close(vw);
            cleanup = []; %#ok<NASGU>  release before stat
            d = dir(tmp);
            testCase.verifyNotEmpty(d, ...
                sprintf("expected output file %s", tmp));
            if ~isempty(d)
                testCase.verifyGreaterThan(d.bytes, 0);
                delete(tmp);
            end
        end
    end
end

function safe_close_writer(vw)
    try
        if isa(vw, 'VideoWriter')
            close(vw);
        end
    catch
        % already closed
    end
end

function close_if_valid(h)
    try
        if isgraphics(h, 'figure')
            close(h);
        end
    catch
    end
end

function [result, target, opts] = make_inputs()
    N = 30;
    t = linspace(0, 0.3, N).';
    head = [0.5 * sin(2 * pi * t / 0.3), ...
            0.5 * cos(2 * pi * t / 0.3), ...
            0.2 + 0.05 * t];
    butt = head - repmat([0 0 1.0], N, 1);
    quat = repmat([1 0 0 0], N, 1);

    target = struct( ...
        "time", t, "butt", butt, "clubhead", head, ...
        "club_quat", quat, "impact_idx", uint32(round(N / 2)), ...
        "source", struct("filename", "synth.xlsx", "format", "xlsx", ...
                         "subject_id", "synth", "trial_id", "1", ...
                         "sha256", repmat('0', 1, 64)));

    sim_out = struct( ...
        "time", t, ...
        "r_butt", butt, ...
        "r_clubhead", head, ...
        "q_club", quat, ...
        "solver_status", "success");

    result = struct("sim_out", sim_out, "coefficients", zeros(7 * 6, 1));
    opts = default_viz_options();
    opts.visible = "off";
end
