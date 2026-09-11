classdef test_capture_fit_sim_options < matlab.unittest.TestCase
    methods (Test)
        function rejectsInvalidHorizon(testCase)
            testCase.verifyError(@() capture_fit_sim_options(0), ...
                'MATLAB:validators:mustBePositive');
            testCase.verifyError(@() capture_fit_sim_options(Inf), ...
                'MATLAB:validators:mustBeFinite');
        end

        function usesIndependentReplayDefaults(testCase)
            opts = capture_fit_sim_options(653/360);
            testCase.verifyEqual(opts.simulation_time, 653/360);
            testCase.verifyFalse(opts.fast_restart);
            testCase.verifyFalse(opts.use_cache);
            testCase.verifyEqual(opts.model_name, "GolfSwing3D_Kinetic");
        end

        function torqueGateStaysEnabledPastOneSecond(testCase)
            [~, name] = fileparts(tempname);
            model = ['capture_gate_' name];
            new_system(model);
            testCase.addTeardown(@() bdclose(model));
            ws = get_param(model, 'ModelWorkspace');
            assignin(ws, 'KillswitchInitialValue', 1);
            assignin(ws, 'KillswitchFinalValue', 0);
            assignin(ws, 'KillswitchStepTime', 1);
            assignin(ws, 'HipInputXA', 0);
            add_block('simulink/Sources/Step', [model '/Gate'], ...
                'Time', 'KillswitchStepTime', ...
                'Before', '[KillswitchInitialValue 0 0]', ...
                'After', '[KillswitchFinalValue 0 0]', 'SampleTime', '0.01');
            add_block('simulink/Sinks/To Workspace', [model '/Position'], ...
                'VariableName', 'CHGlobalPosition', 'SaveFormat', 'Array');
            add_line(model, 'Gate/1', 'Position/1');
            set_param(model, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.01');
            opts = capture_fit_sim_options(1.81);
            opts.model_name = model;
            opts.joint_names = "HipInputX";
            opts.solver = "FixedStepDiscrete";
            opts.sample_rate = 100;
            opts.verbosity = "Silent";
            actual = simulate_with_coefficients(zeros(7,1), opts);
            testCase.verifyGreaterThan(actual.time(end), 1.8);
            testCase.verifyEqual(actual.r_clubhead(:,1), ones(size(actual.time)));
            testCase.verifyEqual(getVariable(ws, 'KillswitchFinalValue'), 0);
        end
    end
end
