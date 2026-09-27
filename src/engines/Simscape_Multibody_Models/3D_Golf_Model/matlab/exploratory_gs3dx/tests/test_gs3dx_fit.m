classdef test_gs3dx_fit < matlab.unittest.TestCase
%TEST_GS3DX_FIT  GS3DX_Fit: segment lengths from the capture, whole-body IK (#10979).
%
%   The capture's joint-centre lengths are consistent (rigid segments keep
%   their length), they map onto the model length variables, the model is
%   re-pointed at Fit* variables without adding blocks, the model's own
%   forward kinematics reproduces the data lengths, the hand-on-grip
%   geometry comes from the capture, and the whole-body IK tracks the
%   capture out of sample.

    properties
        info struct
        cap struct
        jc struct
        mdl char
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.cap = gs3dx_capture_markers();
            testCase.jc = gs3dx_capture_joint_centres(testCase.cap);
            testCase.mdl = char(gs3dx_names().variants.fit);
            if ~isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx']))
                gs3dx_build_fit(testCase.info, jc=testCase.jc);
            end
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function joint_centre_lengths_are_rigid(testCase)
            % A proxy that tracks a rigid segment keeps its length: SD under
            % 3 cm and under 10% of the median, for every segment used.
            l = testCase.jc.lengths;
            for f = ["thigh_L", "thigh_R", "shank_L", "shank_R", "upper_arm_L", ...
                    "forearm_L", "forearm_R", "shoulders", "pelvis_to_shoulders"]
                testCase.verifyLessThan(l.(f)(2), 0.03, f);
                testCase.verifyLessThan(l.(f)(2), 0.1 * l.(f)(1), f);
            end
        end

        function gap_filled_samples_are_marked(testCase)
            % The club-head cluster drops out after impact (DATA_AUDIT.md);
            % those filled samples must be flagged so the IK ignores them.
            g = testCase.jc.gap;
            n = size(testCase.jc.pelvis, 2);
            for f = fieldnames(g).'
                testCase.verifySize(g.(f{1}), [1 n], f{1});
            end
            testCase.verifyTrue(all(g.club_head(519:551)), 'post-impact club-head gap (frames 519-551)');
            testCase.verifyFalse(g.club_head(testCase.jc.impact_frame), 'the club head is measured at impact');
        end

        function lengths_map_onto_the_model_variables(testCase)
            fit = gs3dx_fit_lengths(testCase.jc);
            l = testCase.jc.lengths;
            in = 0.0254;
            testCase.verifyEqual(2 * fit.vars.FitHubtoSLength * in, l.shoulders(1), 'RelTol', 1e-12);
            testCase.verifyEqual(fit.vars.FitUpperArmLength * in, l.upper_arm_L(1), 'RelTol', 1e-12);
            testCase.verifyEqual((fit.vars.FitLowerTorsoLength + fit.vars.FitUpperTorsoLength) * in, ...
                l.pelvis_to_shoulders(1), 'RelTol', 1e-12);
            testCase.verifyEqual(fit.vars.ShankLength, (l.shank_L(1) + l.shank_R(1)) / 2, 'RelTol', 1e-12);
            % A plausible adult: every length within 25% of the de Leva /
            % original model value it replaces.
            ref = struct('FitHubtoSLength', 8, 'FitUpperArmLength', 12, 'FitLowerArmLength', 11, ...
                'FitLowerTorsoLength', 11, 'FitUpperTorsoLength', 11, 'ThighLength', 0.4365, 'ShankLength', 0.4437);
            for f = fieldnames(ref).'
                testCase.verifyEqual(fit.vars.(f{1}), ref.(f{1}), 'RelTol', 0.25, f{1});
            end
        end

        function model_reads_the_fit_variables(testCase)
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            fit = gs3dx_fit_lengths(testCase.jc);
            for f = fieldnames(fit.vars).'
                testCase.verifyEqual(ws.getVariable(f{1}), fit.vars.(f{1}), 'RelTol', 1e-12, f{1});
            end
            solids = find_system(testCase.mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'ReferenceBlock', 'sm_lib/Body Elements/Cylindrical Solid');
            old = ["LowerTorsoLength", "UpperTorsoLength", "HubtoSLength", "UpperArmLength", "LowerArmLength", ...
                "WristStandoffLength"];
            for k = 1:numel(solids)
                expr = string(get_param(solids{k}, 'CylinderLength'));
                testCase.verifyEmpty(regexp(expr, "(?<!Fit)(" + strjoin(old, "|") + ")", 'once'), solids{k});
            end
        end

        function grip_reads_the_fit_variables(testCase)
            % The grip lengths are Fit* variables, the butt-to-shaft length
            % stays 10.5 in, and the lead standoff is flipped: its wrist
            % frame sits on the bottom curve, the trail one on the top.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            g = [testCase.mdl '/Grip/'];
            expr = struct('butt', {{'2.5" Butt', 'FitButtToLeadHand'}}, ...
                'top', {{'1.5" Top', '0.5*FitHandSpacing'}}, 'bottom', {{'1.5" Bottom', '0.5*FitHandSpacing'}}, ...
                'ghost', {{'GhostGripSegment', 'FitGripToShaft'}}, ...
                'lead', {{'LHandStandoff', 'FitLeftWristStandoff'}}, 'trail', {{'RHandStandoff', 'FitRightWristStandoff'}});
            for f = fieldnames(expr).'
                e = expr.(f{1});
                testCase.verifyEqual(get_param([g e{1}], 'CylinderLength'), e{2}, e{1});
            end
            v = @(n) ws.getVariable(n);
            testCase.verifyEqual(v('FitButtToLeadHand') + v('FitHandSpacing') + v('FitGripToShaft'), 10.5, 'AbsTol', 1e-12);
            wrist = @(blk, name) regexp(get_param([g blk], 'SerializedFrames'), ...
                ['<Name>' name '</Name><Origin><Source>GeometricFeature</Source><FeatureName>([a-z ]+)<'], 'tokens', 'once');
            testCase.verifyEqual(wrist('LHandStandoff', 'Left Wrist'), {'bottom curve'}, 'lead standoff flipped');
            testCase.verifyEqual(wrist('RHandStandoff', 'Right Wrist'), {'top curve'}, 'trail standoff as GS3DX_Golfer');
        end

        function fit_adds_no_blocks(testCase)
            src = char(gs3dx_names().variants.golfer);
            load_system(src);
            testCase.addTeardown(@() close_system(src, 0));
            count = @(m) numel(find_system(m, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
            testCase.verifyEqual(count(testCase.mdl), count(src));
        end

        function whole_body_ik_generalizes_to_the_downswing(testCase)
            % Marker offsets calibrated on the backswing only, then held
            % fixed while the model's own kinematics (KinematicsSolver, grip
            % loop closed) is fitted to 14 joint centres over the last
            % 0.25 s before impact: an out-of-sample check.  The bounds pin
            % the 2026-09-27 result with the fitted grip (docs/FIT.md).
            % GS3DX_FIT_GRIP on that IK must return the model's own grip
            % variables (a fixed point of the estimate).
            f0 = testCase.jc.impact_frame;
            ik = gs3dx_whole_body_ik(testCase.jc, frames=f0 - 90:6:f0, calibration_frames=1:15:f0 - 90);
            testCase.verifyEqual(ik.status, ones(size(ik.status)), 'grip loop closed in every frame');
            worst = @(pat) max(ik.residual(contains(ik.names, pat), :), [], 'all');
            fprintf('out of sample: rms max %.1f mm; legs %.1f, wrists %.1f, club head %.1f mm\n', ...
                1000 * max(ik.rms), 1000 * worst(["pelvis", "hip", "knee", "ankle"]), ...
                1000 * worst("wrist"), 1000 * worst("club_head"));
            testCase.verifyLessThan(worst(["pelvis", "hip", "knee", "ankle"]), 0.03, 'pelvis and legs');
            testCase.verifyLessThan(worst("wrist"), 0.035, 'wrists (fitted grip)');
            testCase.verifyLessThan(worst("club_head"), 0.04, 'club head');
            testCase.verifyLessThan(max(ik.rms), 0.025, sprintf('rms up to %.1f mm', 1000 * max(ik.rms)));
            off = struct2cell(ik.offsets);
            testCase.verifyLessThan(max(vecnorm([off{:}])), 0.06, 'marker offsets stay anatomical');

            grip = gs3dx_fit_grip(testCase.cap, ik);
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            for f = fieldnames(grip.vars).'
                fprintf('grip %s: model %.3f, re-estimated %.3f in\n', f{1}, ws.getVariable(f{1}), grip.vars.(f{1}));
                testCase.verifyEqual(grip.vars.(f{1}), ws.getVariable(f{1}), 'AbsTol', 0.25, f{1});
            end
            testCase.verifyLessThan([grip.sphere.rms], 0.015, 'wrist markers move on spheres about the wrist centres');
        end
    end
end
