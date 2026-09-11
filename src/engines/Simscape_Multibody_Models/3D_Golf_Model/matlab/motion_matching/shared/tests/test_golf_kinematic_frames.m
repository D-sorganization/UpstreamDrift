classdef test_golf_kinematic_frames < matlab.unittest.TestCase
    methods (Test)
        function framesMatchIndependentForwardReplay(testCase)
            shared = fileparts(fileparts(mfilename('fullpath')));
            engine = fileparts(fileparts(shared));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(engine, 'src', 'model')));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(engine, 'src', 'functions'), 'IncludingSubfolders', true));
            load_system('GolfSwing3D_Kinetic');
            testCase.addTeardown(@() bdclose('GolfSwing3D_Kinetic'));
            [ks, schema] = build_golf_kinematics();
            addTargetVariables(ks, schema.q_ids);
            addOutputVariables(ks, schema.frame_ids);
            testCase.verifyTrue(ismember("Hip", string({schema.frames.name})));
            addOutputVariables(ks, schema.rotation_ids);
            opts = capture_fit_sim_options(0.02);
            opts.retain_raw_output = true;
            opts.verbosity = "Silent";
            replay = simulate_with_coefficients(zeros(7*numel(schema.coordinate_names),1), opts);
            [found, columns] = ismember(schema.coordinate_names, replay.joint_names);
            testCase.assertTrue(all(found));
            for index = [1 11 21]
                [actual, status, targets] = solve(ks, replay.q(index,columns)');
                testCase.assertEqual(status, 1);
                testCase.assertTrue(all(targets));
                splitIndex = 3*numel(schema.frames);
                positions = reshape(actual(1:splitIndex), 3, [])';
                rotations = intrinsic_xyz_to_rotm(reshape(actual(splitIndex+1:end),3,[])');
                for f = 1:numel(schema.frames)
                    if strcmp(schema.frames(f).name, 'Hip')
                        % HipGlobalPosition measures the fixed joint BASE, not
                        % the moving follower. Use the three measured sliding
                        % coordinates resolved through the native base tilt.
                        workspace = get_param(schema.model_name, 'ModelWorkspace');
                        planeTilt = workspace.getVariable('PlaneTilt');
                        if isa(planeTilt, 'Simulink.Parameter')
                            planeTilt = planeTilt.Value;
                        end
                        baseRotation = intrinsic_xyz_to_rotm( ...
                            [deg2rad(planeTilt) 0 0]);
                        [~, translationColumns] = ismember( ...
                            ["TranslationInputX" "TranslationInputY" "TranslationInputZ"], ...
                            replay.joint_names);
                        expected = replay.q(:,translationColumns)*baseRotation';
                    else
                        signal = replay.raw_output.CombinedSignalBus;
                        for key = split(string(schema.frames(f).source), '.')'
                            signal = signal.(key);
                        end
                        expected = resample_logged_signal(signal, replay.time, 3, []);
                    end
                    testCase.verifyEqual(positions(f,:), expected(index,:), ...
                        'AbsTol', 1e-8, schema.frames(f).name);
                    if strlength(string(schema.frames(f).rotation_source)) > 0
                        signal = replay.raw_output.CombinedSignalBus;
                        for key = split(string(schema.frames(f).rotation_source), '.')'
                            signal = signal.(key);
                        end
                        measured = resample_logged_signal(signal, replay.time, 9, []);
                        expectedRotation = reshape(measured(index,:), 3, 3);
                        testCase.verifyEqual(rotations(:,:,f), expectedRotation, ...
                            'AbsTol', 1e-8, schema.frames(f).name);
                    end
                end
            end
        end
    end
end
