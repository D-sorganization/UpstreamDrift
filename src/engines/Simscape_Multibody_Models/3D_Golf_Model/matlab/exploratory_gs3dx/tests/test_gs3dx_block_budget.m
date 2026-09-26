classdef test_gs3dx_block_budget < matlab.unittest.TestCase
%TEST_GS3DX_BLOCK_BUDGET  Unit tests for gs3dx_block_budget and license probe.
%
%   Run:  results = runtests('test_gs3dx_block_budget') (from exploratory_gs3dx/tests)

    properties
        info struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(root);
            testCase.info = gs3dx_setup();
        end
    end

    methods (Test)
        function small_in_memory_model_exact_budget(testCase)
            % Test 1: In-memory model with known block count is exact.
            name = 'GS3DX_test_small_exact';
            if bdIsLoaded(name)
                close_system(name, 0);
            end
            new_system(name);
            c = onCleanup(@() close_system(name, 0));

            add_block('simulink/Sources/Constant', [name '/Const']);
            add_block('simulink/Math Operations/Gain', [name '/Gain1']);
            add_block('simulink/Math Operations/Gain', [name '/Gain2']);
            add_block('simulink/Sinks/Terminator', [name '/Term']);

            add_line(name, 'Const/1', 'Gain1/1');
            add_line(name, 'Gain1/1', 'Gain2/1');
            add_line(name, 'Gain2/1', 'Term/1');

            report = gs3dx_block_budget(name);

            % In Simulink, Terminator is a virtual block (Virtual='on').
            % Therefore, nonvirtual blocks are Const, Gain1, Gain2 = 3.
            testCase.verifyEqual(report.nonvirtual_total, 3, ...
                'Expected exactly 3 nonvirtual blocks (Const, 2 Gains).');
            testCase.verifyEqual(report.converter_internal, 0, ...
                'Expected 0 converter internal blocks.');
            testCase.verifyEqual(report.top_level_equivalent, 3, ...
                'Expected top-level equivalent to match nonvirtual total.');
            testCase.verifyEqual(report.simscape_blocks, 0, ...
                'Expected 0 Simscape blocks.');
            testCase.verifyTrue(istable(report.by_type));
            testCase.verifyEqual(height(report.by_type), 2);

            gain_row = report.by_type(report.by_type.BlockType == "Gain", :);
            testCase.verifyEqual(gain_row.Count, 2);
        end

        function small_in_memory_model_with_converter(testCase)
            % Test 2: In-memory model with converters verifies converter counts.
            name = 'GS3DX_test_small_conv';
            if bdIsLoaded(name)
                close_system(name, 0);
            end
            new_system(name);
            c = onCleanup(@() close_system(name, 0));

            add_block('simulink/Sources/Constant', [name '/Const']);
            add_block('nesl_utility/Solver Configuration', [name '/SC']);
            add_block('nesl_utility/Simulink-PS Converter', [name '/SPS']);
            add_block('nesl_utility/PS-Simulink Converter', [name '/PSS']);
            add_block('simulink/Sinks/Terminator', [name '/Term']);

            add_line(name, 'Const/1', 'SPS/1');
            add_line(name, 'PSS/1', 'Term/1');

            report = gs3dx_block_budget(name);

            % Nonvirtual blocks: 1 Const + 1 SC + 1 SPS(internal PMIOPort) + 1 PSS(internal PMIOPort) = 4
            % (Terminator is virtual in Simulink)
            testCase.verifyEqual(report.nonvirtual_total, 4);
            testCase.verifyEqual(report.converter_internal, 2);
            testCase.verifyEqual(report.top_level_equivalent, 4);
            % SC, SPS, PSS all reference nesl_utility
            testCase.verifyEqual(report.simscape_blocks, 3);
        end

        function json_export_is_valid(testCase)
            % Test 3: json_path option produces valid JSON with correct contents.
            name = 'GS3DX_test_small_json';
            if bdIsLoaded(name)
                close_system(name, 0);
            end
            new_system(name);
            c_mdl = onCleanup(@() close_system(name, 0));

            add_block('simulink/Sources/Constant', [name '/Const']);
            add_block('simulink/Math Operations/Gain', [name '/Gain']);
            add_block('simulink/Sinks/Terminator', [name '/Term']);
            add_line(name, 'Const/1', 'Gain/1');
            add_line(name, 'Gain/1', 'Term/1');

            json_file = fullfile(tempdir, 'gs3dx_test_budget_temp.json');
            if isfile(json_file)
                delete(json_file);
            end
            c_file = onCleanup(@() delete(json_file));

            report = gs3dx_block_budget(name, json_path=json_file);

            testCase.verifyTrue(isfile(json_file), 'JSON file was not created.');
            raw_text = fileread(json_file);
            testCase.verifyNotEmpty(raw_text);

            decoded = jsondecode(raw_text);
            testCase.verifyEqual(decoded.nonvirtual_total, report.nonvirtual_total);
            testCase.verifyEqual(decoded.converter_internal, report.converter_internal);
            testCase.verifyEqual(decoded.top_level_equivalent, report.top_level_equivalent);
            testCase.verifyEqual(decoded.simscape_blocks, report.simscape_blocks);
        end

        function baseline_budget_sanity(testCase)
            % Test 4: GS3DX_Baseline returns nonvirtual_total > 0 and converter_internal <= nonvirtual_total.
            report = gs3dx_block_budget('GS3DX_Baseline');

            testCase.verifyGreaterThan(report.nonvirtual_total, 0, ...
                'Baseline model should have positive nonvirtual block count.');
            testCase.verifyLessThanOrEqual(report.converter_internal, report.nonvirtual_total, ...
                'Converter internal blocks must not exceed nonvirtual total.');
            testCase.verifyGreaterThan(report.top_level_equivalent, 0);
            testCase.verifyGreaterThan(report.simscape_blocks, 0);
            testCase.verifyTrue(istable(report.by_type));
            testCase.verifyEqual(sum(report.by_type.Count), report.nonvirtual_total, ...
                'Sum of by_type counts must equal nonvirtual_total.');
        end

        function license_probe_fast_smoke(testCase)
            % Test 5: Smoke test license probe on small counts without throwing.
            results = gs3dx_license_limit_probe(gain_counts=3, conv_counts=2);
            testCase.verifyTrue(istable(results));
            testCase.verifyEqual(height(results), 2);
            testCase.verifyTrue(all(results.Status == "Success"));
        end
    end
end
