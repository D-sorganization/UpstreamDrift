classdef test_simulation_input_workspace < matlab.unittest.TestCase
    % Native Simulink regression for the matching wrapper (epic #9921, #9925).
    properties
        ModelName
    end

    methods (TestMethodSetup)
        function createModel(testCase)
            [~, uniqueName] = fileparts(tempname);
            model = ['matching_' uniqueName];
            new_system(model);
            testCase.ModelName = model;
            testCase.addTeardown(@() bdclose(model));
            workspace = get_param(model, 'ModelWorkspace');
            assignin(workspace, 'HipInputXA', Simulink.Parameter(1));
            assignin(workspace, 'MarkerOffset', 0);
            add_block('simulink/Sources/Constant', [model '/Marker'], ...
                'Value', '[HipInputXA+MarkerOffset 0 0]', 'SampleTime', '0.01');
            add_block('simulink/Sinks/To Workspace', [model '/Clubhead'], ...
                'VariableName', 'CHGlobalPosition', 'SaveFormat', 'Array');
            add_block('simulink/Sinks/To Workspace', [model '/Grip'], ...
                'VariableName', 'MPGlobalPosition', 'SaveFormat', 'Array');
            add_line(model, 'Marker/1', 'Clubhead/1');
            add_line(model, 'Marker/1', 'Grip/1');
            set_param(model, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.01', ...
                'ReturnWorkspaceOutputs', 'on');
        end
    end

    methods (Test)
        function fastRestartCanBeDisabledForIndependentReplay(testCase)
            opts = testCase.options();
            testCase.addTeardown(@() set_param(testCase.ModelName, 'FastRestart', 'off'));
            opts.fast_restart = true;
            simulate_with_coefficients(zeros(7,1), opts);
            testCase.verifyEqual(get_param(testCase.ModelName, 'FastRestart'), 'on');
            opts.fast_restart = false;
            replay = simulate_with_coefficients([3; zeros(6,1)], opts);
            testCase.verifyEqual(get_param(testCase.ModelName, 'FastRestart'), 'off');
            testCase.verifyEqual(replay.r_clubhead(:,1), 3*ones(size(replay.time)));
        end

        function coefficientOverridesReachModelWorkspace(testCase)
            opts = testCase.options();
            theta = [5; zeros(6,1)];
            result = simulate_with_coefficients(theta, opts);
            testCase.verifyEqual(result.r_clubhead(:,1), ...
                5 * ones(numel(result.time),1), 'AbsTol', 1e-12);
            workspace = get_param(testCase.ModelName, 'ModelWorkspace');
            original = getVariable(workspace, 'HipInputXA');
            testCase.verifyEqual(original.Value, 1);
        end

        function requestedRawEvidenceIsRetained(testCase)
            opts = testCase.options();
            opts.retain_raw_output = true;
            result = simulate_with_coefficients(zeros(7,1), opts);
            testCase.verifyTrue(isfield(result, 'raw_output'));
            testCase.verifyClass(result.raw_output, 'Simulink.SimulationOutput');
        end

        function defaultOutputDoesNotRetainRawEvidence(testCase)
            result = simulate_with_coefficients(zeros(7,1), testCase.options());
            testCase.verifyFalse(isfield(result, 'raw_output'));
        end

        function initialAndGeometryOverridesReachModelWorkspace(testCase)
            opts = testCase.options();
            opts.input_overrides = struct('MarkerOffset', 2);
            result = simulate_with_coefficients([1; zeros(6,1)], opts);
            testCase.verifyEqual(result.r_clubhead(:,1), ...
                3 * ones(numel(result.time),1), 'AbsTol', 1e-12);
        end

        function rawEvidenceBypassesCache(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            opts = testCase.options();
            opts.retain_raw_output = true;
            opts.use_cache = true;
            opts.cache_dir = folder.Folder;
            simulate_with_coefficients(zeros(7,1), opts);
            result = simulate_with_coefficients(zeros(7,1), opts);
            testCase.verifyFalse(result.cache_hit);
            testCase.verifyClass(result.raw_output, 'Simulink.SimulationOutput');
        end
    end

    methods (Access = private)
        function opts = options(testCase)
            opts = default_sim_options();
            opts.model_name = testCase.ModelName;
            opts.joint_names = "HipInputX";
            opts.simulation_time = 0.02;
            opts.fast_restart = false;
            opts.solver = "FixedStepDiscrete";
            opts.use_cache = false;
            opts.verbosity = "Silent";
        end
    end
end
