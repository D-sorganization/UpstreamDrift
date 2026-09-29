classdef test_gimbal_initial_targets < matlab.unittest.TestCase
    methods (Test)
        function shippedPrimitiveUsesInstanceTargets(testCase)
            directory = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'src', 'model');
            model = 'Kinetically_Driven_Gimbal_Joint';
            load_system(fullfile(directory, [model '.slx']));
            testCase.addTeardown(@() bdclose(model));
            primitive = [model '/Kinetically Driven'];
            for axis = 'XYZ'
                for quantity = ["Position", "Velocity"]
                    key = "R" + lower(string(axis)) + quantity + "TargetValue";
                    expected = "Start" + quantity + string(axis);
                    testCase.verifyEqual(string(get_param(primitive, key)), expected);
                end
            end
        end

        function migrationIsIdempotent(testCase)
            directory = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'src', 'model');
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            model = 'Kinetically_Driven_Gimbal_Joint';
            file = fullfile(folder.Folder, [model '.slx']);
            copyfile(fullfile(directory, [model '.slx']), file);
            load_system(file);
            set_param([model '/Kinetically Driven'], 'RxPositionTargetValue', 'LSStartPositionX');
            save_system(model, file);
            bdclose(model);
            first = repair_gimbal_initial_targets(file);
            second = repair_gimbal_initial_targets(file);
            testCase.verifyTrue(first.changed);
            testCase.verifyFalse(second.changed);
            testCase.verifyEqual(second.before, second.after);
        end
    end
end
