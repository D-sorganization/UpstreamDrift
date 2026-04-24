classdef test_dataset_generator_config_compatibility < matlab.unittest.TestCase
    % TEST_DATASET_GENERATOR_CONFIG_COMPATIBILITY Regression coverage for the
    % bundled Simscape golf dataset generator configuration path.

    methods (TestClassSetup)
        function addDatasetGeneratorPaths(testCase)
            test_file = mfilename('fullpath');
            tests_root = fileparts(test_file);
            matlab_root = fileparts(tests_root);

            scripts_dir = fullfile(matlab_root, 'src', 'scripts', 'dataset_generator');
            functions_dir = fullfile(matlab_root, 'src', 'functions', 'dataset_generator');
            model_dir = fullfile(matlab_root, 'src', 'model');

            addpath(scripts_dir);
            addpath(genpath(functions_dir));
            addpath(model_dir);

            testCase.TestData.output_dir = tempname;
            mkdir(testCase.TestData.output_dir);
        end
    end

    methods (TestClassTeardown)
        function removeTestOutput(testCase)
            if isfield(testCase.TestData, 'output_dir') && exist(testCase.TestData.output_dir, 'dir')
                rmdir(testCase.TestData.output_dir, 's');
            end
        end
    end

    methods (Test)
        function defaultConfigTargetsBundledModel(testCase)
            config = createSimulationConfig();

            testCase.verifyEqual(config.model_name, 'GolfSwing3D_Kinetic');
            testCase.verifyTrue(exist(config.model_path, 'file') == 2, ...
                sprintf('Expected model file to exist: %s', config.model_path));
            testCase.verifyTrue(isfield(config, 'verbose'));
            testCase.verifyTrue(config.verbose);
        end

        function ensureEnhancedConfigBackfillsVerboseFromVerbosity(testCase)
            silent_config = ensureEnhancedConfig(struct('verbosity', 'Silent'));
            testCase.verifyFalse(silent_config.verbose);

            verbose_config = ensureEnhancedConfig(struct('verbosity', 'Verbose'));
            testCase.verifyTrue(verbose_config.verbose);
        end

        function processSimulationOutputHandlesMissingVerboseField(testCase)
            config = createSimulationConfig();
            config.output_folder = testCase.TestData.output_dir;
            config = rmfield(config, 'verbose');

            result = processSimulationOutput(1, config, struct(), false);

            testCase.verifyFalse(result.success);
            testCase.verifyTrue(isfield(result, 'error'));
            testCase.verifyTrue(contains(result.error, 'No data extracted'));
        end
    end
end
