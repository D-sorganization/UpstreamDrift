classdef test_gs3dx_fit_legs < matlab.unittest.TestCase
%TEST_GS3DX_FIT_LEGS  Leg servo references from the capture, GS3DX_FitLegs (#10979).
%
%   GS3DX_LEG_REFERENCE on a whole-body IK of the backswing and downswing
%   (every 18th frame to impact, 20 Hz, filtered at 6 Hz) puts the model's feet exactly on the
%   measured, levelled foot path under the model's pelvis path, with the
%   knees near the capture; GS3DX_FitLegs plays a reference through its leg
%   servo with no extra block, and stands on it from rest.

    properties
        info struct
        ref struct
        mdl char
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            cap = gs3dx_capture_markers();
            jc = gs3dx_capture_joint_centres(cap);
            f0 = jc.impact_frame;
            ik = gs3dx_whole_body_ik(jc, frames=1:18:f0, calibration_frames=1:15:f0 - 90);
            testCase.ref = gs3dx_leg_reference(ik, jc, cap, cutoff_hz=6);
            testCase.mdl = char(gs3dx_names().variants.fit_legs);
            if ~isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx']))
                gs3dx_build_fit_legs(testCase.info, testCase.ref);
            end
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function reference_puts_the_feet_on_the_foot_path(testCase)
            % Six angles for six foot-pose numbers: GS3DX_LEG_FK of the
            % reference under the filtered pelvis reproduces the foot path.
            r = testCase.ref;
            ws = get_param(r.model, 'ModelWorkspace');
            sides = 'LR';
            for k = 1:2
                P = sides(k);
                geom = struct('mount_R', ws.getVariable([P 'HipMountRotation']), ...
                    'mount_p', ws.getVariable([P 'HipMountOffset']), ...
                    'thigh', ws.getVariable('ThighLength'), 'shank', ws.getVariable('ShankLength'));
                worst = 0;
                for i = 1:numel(r.frames)
                    [R, p] = gs3dx_leg_fk(geom, r.pelvis_R(:, :, i), r.pelvis_p(:, i), r.q((k - 1) * 6 + (1:6), i));
                    worst = max([worst, norm(p - r.feet.(P).p(:, i)), norm(R - r.feet.(P).R(:, :, i))]);
                end
                testCase.verifyLessThan(worst, 1e-6, [P ' foot off its path']);
            end
        end

        function reference_stays_near_the_capture(testCase)
            % The ankle target moves toward the hip only where the lead knee
            % is straight, the feet are levelled by millimetres, the torsion
            % is a constant over the still address, and the knees stay near
            % the capture knees.  The bounds pin the 2026-09-27 result
            % (docs/FIT.md).
            r = testCase.ref;
            fprintf('clamp %.1f mm; foot shift %s mm; torsion %s deg; knee median %s, p95 %s mm\n', ...
                1000 * max(r.clamp, [], 'all'), mat2str(1000 * r.foot_shift, 3), mat2str(r.torsion, 3), ...
                mat2str(1000 * median(r.knee_error, 2, 'omitnan').', 3), ...
                mat2str(1000 * prctile(r.knee_error, 95, 2).', 3));
            testCase.verifyLessThan(max(r.clamp, [], 'all'), 0.012, 'ankle target moved toward the hip');
            testCase.verifyLessThan(abs(r.foot_shift), 0.012, 'foot levelling');
            testCase.verifyLessThan(abs(r.torsion), 30, 'foot torsion');
            for P = 'LR'
                testCase.verifyLessThan(std(r.torsion_fit.(P).yaw_error), 2, [P ' torsion varies over the address']);
            end
            testCase.verifyLessThan(median(r.knee_error, 2, 'omitnan'), [0.03; 0.035], 'knee median');
            testCase.verifyLessThan(prctile(r.knee_error, 95, 2), [0.065; 0.065], 'knee p95');
        end

        function model_plays_the_reference_through_the_servo(testCase)
            m = testCase.mdl;
            blk = [m '/Lower Body/Leg Torque Commands'];
            testCase.verifyEqual(get_param(blk, 'BlockType'), 'FromWorkspace');
            testCase.verifySubstring(get_param(blk, 'VariableName'), ...
                'LegServoKp(:) .* LegReferenceAngle + LegServoKd(:) .* LegReferenceRate');
            src = char(gs3dx_names().variants.fit);
            if bdIsLoaded(src)
                close_system(src, 0);   % the KinematicsSolver leaves it compiled (967)
            end
            load_system(src);
            testCase.addTeardown(@() close_system(src, 0));
            count = @(x) numel(find_system(x, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
            testCase.verifyEqual(count(m), count(src), 'one block for one');
            ws = get_param(m, 'ModelWorkspace');
            t = ws.getVariable('LegReferenceTime');
            q = ws.getVariable('LegReferenceAngle');
            testCase.verifyEqual(t(1), 0);
            testCase.verifyGreaterThan(diff(t), 0);
            testCase.verifySize(q, [12 numel(t)]);
            testCase.verifySize(ws.getVariable('LegReferenceRate'), [12 numel(t)]);
            start = ws.getVariable('LegReferenceStart');
            testCase.verifyEqual(start.LegAngleReference, q(:, 1));
            for f = fieldnames(start).'
                testCase.verifyEqual(ws.getVariable(f{1}), start.(f{1}), f{1});
            end
        end

        function stands_on_the_reference_from_rest(testCase)
            % The standing test of GS3DX_FullBodyContact and GS3DX_Golfer,
            % with the same bounds, from the reference's first frame.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            c = gs3dx_contact_check(testCase.info, rest=true, model=testCase.mdl, ...
                variables=ws.getVariable('LegReferenceStart'));
            fprintf('support %s; slip %.1f/%.1f, lift %.1f/%.1f, pelvis %.1f mm\n', mat2str(c.support, 3), ...
                1000 * [c.feet.L.slip c.feet.R.slip c.feet.L.lift c.feet.R.lift c.pelvis]);
            testCase.verifyTrue(c.newton.pass, 'contacts and gravity are the only external forces');
            for P = 'LR'
                testCase.verifyLessThan(c.feet.(P).slip, 5e-3, [P ' foot slides']);
                testCase.verifyLessThan(c.feet.(P).lift, 1e-3, [P ' foot lifts']);
            end
            testCase.verifyGreaterThan(c.support(2), 0.9, 'the ground carries the body weight');
        end
    end
end
