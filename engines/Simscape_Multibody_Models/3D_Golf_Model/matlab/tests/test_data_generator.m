classdef test_data_generator < matlab.unittest.TestCase
    % TEST_DATA_GENERATOR Test suite for data_generator.m module
    %
    % This test suite validates the data_generator.m module extracted from
    % Dataset_GUI.m. Tests ensure functionality is preserved during refactoring.
    %
    % Usage:
    %   results = runtests('test_data_generator')
    %   table(results)
    %
    % Test Categories:
    %   - Configuration validation
    %   - Simulation execution (sequential)
    %   - Simulation execution (parallel)
    %   - Error handling
    %   - Integration with Dataset_GUI
    %
    % See also: DATA_GENERATOR, DATASET_GUI

    properties (TestParameter)
        % Test parameter sets for different scenarios
        num_trials = {1, 5, 10}
        execution_mode = {'sequential', 'parallel'}
    end

    methods (TestClassSetup)
        function setupTestEnvironment(testCase)
            % Setup test environment before all tests

            % Add paths
            addpath(genpath('matlab/Scripts'));

            % Verify model exists
            model_name = 'GolfSwing3D_Model';
            if ~exist([model_name '.slx'], 'file') && ~exist([model_name '.mdl'], 'file')
                testCase.assumeFail('Simulink model not found - skipping tests requiring model');
            end

            % Create test output directory
            testCase.TestData.test_output_dir = fullfile(pwd, 'test_output', ...
                ['test_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))]);
            mkdir(testCase.TestData.test_output_dir);

            fprintf('Test output directory: %s\n', testCase.TestData.test_output_dir);
        end
    end

    methods (TestClassTeardown)
        function cleanupTestEnvironment(testCase)
            % Cleanup after all tests

            % Remove test output directory if empty
            if exist(testCase.TestData.test_output_dir, 'dir')
                contents = dir(testCase.TestData.test_output_dir);
                if length(contents) <= 2  % Only . and ..
                    rmdir(testCase.TestData.test_output_dir);
                else
                    fprintf('Test output preserved: %s\n', testCase.TestData.test_output_dir);
                end
            end
        end
    end

    methods (Test)
        %% Configuration Tests

        function testCreateSimulationConfig(testCase)
            % Test configuration creation helper

            config = createSimulationConfig();

            % Verify required fields exist
            testCase.verifyTrue(isfield(config, 'model_name'), 'Config must have model_name');
            testCase.verifyTrue(isfield(config, 'num_simulations'), 'Config must have num_simulations');
            testCase.verifyTrue(isfield(config, 'simulation_time'), 'Config must have simulation_time');
            testCase.verifyTrue(isfield(config, 'output_folder'), 'Config must have output_folder');

            % Verify defaults are reasonable
            testCase.verifyGreaterThan(config.num_simulations, 0, 'num_simulations must be positive');
            testCase.verifyGreaterThan(config.simulation_time, 0, 'simulation_time must be positive');
            testCase.verifyClass(config.model_name, 'char', 'model_name must be string');
        end

        function testConfigValidation(testCase)
            % Test configuration validation

            % Valid config should pass
            config = createSimulationConfig();
            config.output_folder = testCase.TestData.test_output_dir;

            % This will be implemented in data_generator.m
            % testCase.verifyWarningFree(@() validateSimulationConfig(config));

            % Invalid configs should fail
            bad_config = config;
            bad_config.num_simulations = -1;
            % testCase.verifyError(@() validateSimulationConfig(bad_config), 'DataGenerator:InvalidConfig');

            fprintf('Config validation test ready for implementation\n');
        end

        %% Simulation Execution Tests

        function testSequentialSimulation_SingleTrial(testCase)
            % Test single trial sequential simulation

            config = createSimulationConfig();
            config.num_simulations = 1;
            config.output_folder = testCase.TestData.test_output_dir;
            config.execution_mode = 'sequential';

            % This test will work once data_generator.m is extracted
            % [successful_trials, dataset_path] = runSimulation(config);
            % testCase.verifyEqual(successful_trials, 1, 'Single trial should succeed');
            % testCase.verifyTrue(exist(dataset_path, 'dir') > 0, 'Dataset folder should exist');

            fprintf('Sequential simulation test ready for implementation\n');
        end

        function testParallelSimulation_MultiplTrials(testCase)
            % Test parallel simulation with multiple trials

            % Check if parallel computing toolbox available
            if ~license('test', 'Distrib_Computing_Toolbox')
                testCase.assumeFail('Parallel Computing Toolbox not available');
            end

            config = createSimulationConfig();
            config.num_simulations = 5;
            config.output_folder = testCase.TestData.test_output_dir;
            config.execution_mode = 'parallel';

            % This test will work once data_generator.m is extracted
            % [successful_trials, dataset_path] = runSimulation(config);
            % testCase.verifyGreaterThanOrEqual(successful_trials, 1, 'At least one trial should succeed');

            fprintf('Parallel simulation test ready for implementation\n');
        end

        %% Error Handling Tests

        function testInvalidModelName(testCase)
            % Test error handling for invalid model name

            config = createSimulationConfig();
            config.model_name = 'NonExistentModel';
            config.output_folder = testCase.TestData.test_output_dir;

            % Should throw error for non-existent model
            % testCase.verifyError(@() runSimulation(config), 'DataGenerator:ModelNotFound');

            fprintf('Error handling test ready for implementation\n');
        end

        function testInvalidOutputFolder(testCase)
            % Test error handling for invalid output folder

            config = createSimulationConfig();
            config.output_folder = '/invalid/path/that/does/not/exist';

            % Should throw error or create folder
            % testCase.verifyError(@() runSimulation(config), 'DataGenerator:InvalidPath');

            fprintf('Output folder test ready for implementation\n');
        end

        %% Integration Tests

        function testBackwardCompatibility(testCase)
            % Test that data_generator.m works with Dataset_GUI.m

            % Verify Dataset_GUI can call data_generator functions
            % This ensures refactoring doesn't break existing GUI

            fprintf('Backward compatibility test ready for implementation\n');
        end

        function testOutputFormat(testCase)
            % Test that output format matches expected structure

            config = createSimulationConfig();
            config.num_simulations = 1;
            config.output_folder = testCase.TestData.test_output_dir;

            % After extraction, verify output structure
            % [~, dataset_path] = runSimulation(config);
            % files = dir(fullfile(dataset_path, '*.csv'));
            % testCase.verifyGreaterThanOrEqual(length(files), 1, 'Should produce CSV files');

            fprintf('Output format test ready for implementation\n');
        end
    end

    methods (Test, ParameterCombination = 'sequential')
        function testVariousTrialCounts(testCase, num_trials)
            % Test different numbers of trials

            config = createSimulationConfig();
            config.num_simulations = num_trials;
            config.output_folder = fullfile(testCase.TestData.test_output_dir, ...
                sprintf('trials_%d', num_trials));

            % Once extracted:
            % [successful_trials, ~] = runSimulation(config);
            % testCase.verifyGreaterThanOrEqual(successful_trials, 0);
            % testCase.verifyLessThanOrEqual(successful_trials, num_trials);

            fprintf('Trial count test ready: %d trials\n', num_trials);
        end
    end
end

%% Helper Functions for Tests

function config = createSimulationConfig()
    % CREATESIMULATIONCONFIG Create default configuration for testing
    %
    % This helper creates a valid simulation configuration with sensible
    % defaults for testing purposes.
    %
    % Returns:
    %   config - Struct with simulation configuration
    %
    % Example:
    %   config = createSimulationConfig();
    %   config.num_simulations = 10;
    %   [trials, path] = runSimulation(config);

    config = struct();

    % Model settings
    config.model_name = 'GolfSwing3D_Model';
    config.model_path = which([config.model_name '.slx']);
    if isempty(config.model_path)
        config.model_path = which([config.model_name '.mdl']);
    end

    % Simulation settings
    config.num_simulations = 5;
    config.simulation_time = 0.5;  % seconds
    config.sample_rate = 1000;  % Hz

    % Execution settings
    config.execution_mode = 'sequential';  % or 'parallel'
    config.batch_size = 10;
    config.save_interval = 5;

    % Data source settings
    config.use_logsout = true;
    config.use_signal_bus = true;
    config.use_simscape = true;

    % Output settings
    config.output_folder = fullfile(pwd, 'test_output');
    config.folder_name = 'test_dataset';

    % Verbosity
    config.verbosity = 'Normal';  % Silent, Normal, Verbose, Debug

    % Optional settings
    config.enable_animation = false;
    config.capture_workspace = false;
    config.enable_memory_monitoring = false;
    config.enable_checkpoint_resume = false;

    % Torque scenario
    config.torque_scenario = 1;  % 1=Variable, 2=Zero, 3=Constant
    config.coeff_range = 10.0;

    % Coefficient values (will be generated if not provided)
    % config.coefficient_values = [];  % Will be filled by generator
end
