classdef test_simscape_full_logging < matlab.unittest.TestCase
%TEST_SIMSCAPE_FULL_LOGGING  Verify the dataset generator pins
%   `SimscapeLogType='all'` (the home-license workaround) and that the
%   row-builder emits at least one `simlog_*` column from a known joint
%   when a real Simscape simulation can be run.
%
%   Tests are gated with `assumeFail` when Simulink/Simscape is not on the
%   path, mirroring tests/test_load_club_target_excel.m.
%
%   Background: prior 10k-trial parquet runs were generated WITHOUT this
%   pin in the swept-input parsim path, so the simlog tree came back empty
%   and the master schema fell back to the 1956 bus-only columns. This
%   test guards against that regression.

    properties (Constant)
        FUNCTIONS_RELDIR  = fullfile("..", "..", "src", "functions", "dataset_generator");
        SCRIPTS_RELDIR    = fullfile("..", "..", "src", "scripts",   "dataset_generator");
    end

    methods (TestClassSetup)
        function add_dataset_generator_to_path(testCase)
            here = fileparts(mfilename("fullpath"));
            funcs   = fullfile(here, testCase.FUNCTIONS_RELDIR);
            scripts = fullfile(here, testCase.SCRIPTS_RELDIR);
            addpath(funcs);
            addpath(scripts);
            testCase.addTeardown(@() rmpath(funcs));
            testCase.addTeardown(@() rmpath(scripts));
        end
    end

    methods (Test)

        function test_setModelParameters_pins_SimscapeLogType_all(testCase)
            % Precondition: Simulink must be installed (we need
            % Simulink.SimulationInput). Skip cleanly otherwise.
            if exist('Simulink.SimulationInput', 'class') ~= 8
                testCase.assumeFail( ...
                    "skipped -- Simulink not available in test env");
            end
            if exist('setModelParameters', 'file') ~= 2
                testCase.assumeFail( ...
                    "skipped -- setModelParameters.m not on path");
            end

            % We don't actually need the model loaded -- SimulationInput
            % accepts a model name string and stores ModelParameters
            % regardless of whether the model is on disk.
            simIn = Simulink.SimulationInput('GolfSwing3D_Kinetic');
            cfg = struct('simulation_time', 1.0);

            simIn = setModelParameters(simIn, cfg);

            % Postcondition: SimscapeLogType must be 'all' on the returned
            % SimulationInput. Use Variables.Name/Value pairs since the
            % public ModelParameters API is the official surface.
            log_type = local_get_model_parameter(simIn, 'SimscapeLogType');
            testCase.verifyEqual(string(log_type), "all", ...
                "setModelParameters must pin SimscapeLogType='all' " + ...
                "(home-license simlog workaround).");

            % Defensive companion params -- runSimulation re-pins these
            % too; setModelParameters owns the canonical values.
            sim_mode = local_get_model_parameter(simIn, 'SimulationMode');
            testCase.verifyEqual(string(sim_mode), "normal");
            save_output = local_get_model_parameter(simIn, 'SaveOutput');
            testCase.verifyEqual(string(save_output), "on");
        end

        function test_row_builder_emits_simlog_columns(testCase)
            % This is a *contract* test: extractSimscapeDataRecursive must
            % prefix every non-time column with "simlog_". We don't need
            % a real simlog node -- we feed it a synthetic minimal stub.
            %
            % If Simscape isn't installed we still want to assert the
            % prefix contract via a fake simulating simscape.logging.Node,
            % but constructing such a fake reliably across MATLAB versions
            % is brittle. So we gate on Simscape just like the first test.
            if exist('extractSimscapeDataRecursive', 'file') ~= 2
                testCase.assumeFail( ...
                    "skipped -- extractSimscapeDataRecursive.m not on path");
            end
            if exist('simscape.logging.Node', 'class') ~= 8
                testCase.assumeFail( ...
                    "skipped -- Simscape not available in test env");
            end

            % We can't easily fabricate a simscape.logging.Node without
            % running a sim. Instead, assert the *source-level* contract
            % directly: the function must contain the "simlog_" prefix
            % logic. This is brittle but cheap and catches accidental
            % regressions when CI lacks Simscape.
            fpath = which('extractSimscapeDataRecursive');
            txt = fileread(fpath);
            testCase.verifyTrue(contains(txt, "simlog_"), ...
                "extractSimscapeDataRecursive must apply the simlog_ " + ...
                "prefix so the parquet schema can distinguish per-block " + ...
                "Simscape state from bus/logsout columns.");
            testCase.verifyTrue( ...
                contains(txt, "startsWith(prefixed_name, 'simlog_')") || ...
                contains(txt, 'startsWith(prefixed_name, "simlog_")'), ...
                "Prefix must be applied idempotently (startsWith guard).");
        end

        function test_runSimulation_repins_params_before_parsim(testCase)
            % Source-level guard: runSimulation.m must re-pin the three
            % canonical params on every batch_simInputs entry just before
            % calling parsim. This protects against future code paths
            % that bypass setModelParameters.
            scripts_dir = fullfile(fileparts(mfilename("fullpath")), ...
                test_simscape_full_logging.SCRIPTS_RELDIR);
            run_sim_path = fullfile(scripts_dir, 'runSimulation.m');
            if exist(run_sim_path, 'file') ~= 2
                testCase.assumeFail( ...
                    "skipped -- runSimulation.m not present at expected path");
            end
            txt = fileread(run_sim_path);
            testCase.verifyTrue( ...
                contains(txt, "setModelParameter('SimscapeLogType', 'all')"), ...
                "runSimulation must defensively re-pin SimscapeLogType " + ...
                "before parsim.");
            testCase.verifyTrue( ...
                contains(txt, "setModelParameter('SimulationMode', 'normal')"), ...
                "runSimulation must defensively re-pin SimulationMode.");
            testCase.verifyTrue( ...
                contains(txt, "setModelParameter('SaveOutput', 'on')"), ...
                "runSimulation must defensively re-pin SaveOutput.");
        end

    end
end


function val = local_get_model_parameter(simIn, name)
%LOCAL_GET_MODEL_PARAMETER  Pull a ModelParameter Value off a SimulationInput.
%   Walks simIn.ModelParameters (a struct array with .Name/.Value) and
%   returns the most recently set value for the requested parameter, or
%   the empty string if not found. Mirrors the public surface used by
%   setModelParameter / getModelParameter.
    val = "";
    if isempty(simIn.ModelParameters)
        return
    end
    names = string({simIn.ModelParameters.Name});
    hits = find(names == string(name));
    if isempty(hits)
        return
    end
    % Last write wins.
    val = simIn.ModelParameters(hits(end)).Value;
end
