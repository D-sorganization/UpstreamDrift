classdef test_gs3dx_joint_rom < matlab.unittest.TestCase
%TEST_GS3DX_JOINT_ROM  Human range of motion of the GS3DX golfer (#11158).
%
%   GS3DX_JOINT_ROM is the single table of joint limits; GS3DX_ROM_CHECK
%   measures a joint history against it.  The Human model's references
%   (the motion it is driven with) must stay within the normal human range.

    properties
        info struct
        rom table
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.rom = gs3dx_joint_rom();
        end
    end

    methods (Test)
        function every_key_is_a_joint_of_the_human_model(testCase)
            mdl = char(gs3dx_names().variants.human);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [mdl '.slx'])), 'GS3DX_Human is not built');
            keys = gs3dx_joint_keys(mdl);
            missing = setdiff(testCase.rom.key, keys);
            testCase.verifyEmpty(missing, 'ROM keys that are not joints of GS3DX_Human');
        end

        function the_check_measures_the_anatomical_angle(testCase)
            rom = testCase.rom;
            k = find(rom.joint == "RE");   % trail elbow: flexion = -Rz
            q = nan(height(rom), 3);
            q(k, :) = [-10 -100 -160];
            r = gs3dx_rom_check(rom, q);
            testCase.verifyEqual([r.min_deg(k) r.max_deg(k)], [10 160]);
            testCase.verifyEqual(r.excess_deg(k), 10, 'AbsTol', 1e-12);
            testCase.verifyEqual(r.frames(k), 1);
            testCase.verifyFalse(r.ok(k));
            testCase.verifyTrue(all(r.ok([1:k-1 k+1:end])), 'Unchecked rows pass');
        end

        function the_check_wraps_about_the_neutral(testCase)
            rom = testCase.rom;
            k = find(rom.joint == "LF");   % forearm: neutral 90
            q = nan(height(rom), 2);
            q(k, :) = [90 + 350, 90 - 5];   % -10 and -5 about the neutral
            r = gs3dx_rom_check(rom, q);
            testCase.verifyEqual([r.min_deg(k) r.max_deg(k)], [-10 -5], 'AbsTol', 1e-9);
            testCase.verifyTrue(r.ok(k));
        end

        function a_row_without_neutral_checks_its_arc(testCase)
            rom = testCase.rom;
            k = find(rom.motion == "lead wrist radial deviation");
            q = nan(height(rom), 3);
            q(k, :) = [170 -170 150];   % unwrapped arc 190 -> 150..190 = 40
            r = gs3dx_rom_check(rom, q);
            testCase.verifyEqual(r.span_deg(k), 40, 'AbsTol', 1e-9);
            testCase.verifyTrue(r.ok(k));
            testCase.verifyTrue(isnan(r.min_deg(k)));
        end

        function the_rom_penalty_keeps_the_ik_in_the_human_range(testCase)
            % Every 10th frame from address to impact, each warm-started from
            % the one before (isolated frames stall: docs/FORWARD_DYNAMICS.md);
            % the penalty must bring every bounded joint of GS3DX_Fit into
            % range (1 deg slack for the soft penalty) without costing the fit
            % more than 15 mm RMS.  Measured 2026-09-30: 23.3 mm free,
            % 37.7 mm and 0.71 deg with weight 3.
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            f0 = jc.impact_frame;
            frames = unique([1:10:f0 f0]);
            cal = 1:15:f0 - 90;
            reg = {'posture_weight', 0.01, 'smooth_weight', 0.02, 'backward', false, 'gap_weight', 0.5};
            off = gs3dx_whole_body_ik(jc, frames=1, calibration_frames=cal).offsets;
            free = gs3dx_whole_body_ik(jc, 'frames', frames, 'offsets', off, reg{:});
            held = gs3dx_whole_body_ik(jc, 'frames', frames, 'offsets', off, reg{:}, 'rom_weight', 3);
            testCase.verifyEqual(held.regularization.rom_weight, 3);
            r0 = gs3dx_rom_check(testCase.rom, gs3dx_rom_from_ik(free, testCase.rom));
            r = gs3dx_rom_check(testCase.rom, gs3dx_rom_from_ik(held, testCase.rom));
            cols = {'joint', 'motion', 'min_deg', 'max_deg', 'excess_deg'};
            testCase.log(1, sprintf('Without the penalty:\n%s\nWith it:\n%s', ...
                formattedDisplayText(r0(~r0.ok, cols)), formattedDisplayText(r(~r.ok, cols))));
            bounded = ~isnan(r.min_deg);
            testCase.verifyLessThanOrEqual(r.excess_deg(bounded), 1, formattedDisplayText(r(bounded & r.excess_deg > 1, cols)));
            testCase.verifyLessThanOrEqual(mean(held.rms), mean(free.rms) + 0.015, ...
                sprintf('RMS %.1f mm with the penalty, %.1f mm without', 1000 * mean(held.rms), 1000 * mean(free.rms)));
        end

        function human_references_stay_in_the_human_range(testCase)
            mdl = char(gs3dx_names().variants.human);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [mdl '.slx'])), 'GS3DX_Human is not built');
            load_system(mdl);
            testCase.addTeardown(@() close_system(mdl, 0));
            q = gs3dx_rom_reference(get_param(mdl, 'ModelWorkspace'), testCase.rom);
            r = gs3dx_rom_check(testCase.rom, q);
            checked = ~isnan(r.excess_deg);
            testCase.verifyGreaterThanOrEqual(nnz(checked), 20, 'Too few references found');
            bad = r(~r.ok, :);
            testCase.log(1, formattedDisplayText(r(:, {'joint', 'motion', 'min_deg', 'max_deg', 'span_deg', 'excess_deg', 'frames'})));
            testCase.verifyEmpty(bad.motion, sprintf('Out of the human range:\n%s', ...
                formattedDisplayText(bad(:, {'joint', 'motion', 'min_deg', 'max_deg', 'span_deg', 'excess_deg', 'frames'}))));
        end
    end
end
