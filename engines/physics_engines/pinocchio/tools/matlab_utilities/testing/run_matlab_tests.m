function results = run_matlab_tests()
    % RUN_MATLAB_TESTS Comprehensive MATLAB test runner with quality integration
    %
    % This function runs all MATLAB tests and integrates with the project's
    % quality control system. It follows the requirements from .cursorrules.md.
    %
    % Outputs:
    %   results - Struct containing test results and quality metrics
    %            .tests_run - Number of tests executed
    %            .tests_passed - Number of tests that passed
    %            .tests_failed - Number of tests that failed
    %            .quality_checks - Quality check results
    %            .timestamp - ISO 8601 timestamp
    %            .summary - Summary string of results

    fprintf('🧪 Starting comprehensive MATLAB test suite...\n');

    % Initialize results structure
    results = struct();
    results.tests_run = 0;
    results.tests_passed = 0;
    results.tests_failed = 0;
    results.quality_checks = struct();
    results.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');
    results.errors = {};

    % 1. Check MATLAB Unit Testing Framework availability
    if ~exist('matlab.unittest.TestRunner', 'class')
        error_msg = 'MATLAB Unit Testing Framework not available';
        fprintf('❌ %s\n', error_msg);
        results.errors{end+1} = error_msg;
        results.summary = '❌ MATLAB testing framework not available';
        return;
    end

    fprintf('✅ MATLAB Unit Testing Framework available\n');

    % 2. Run quality checks first
    fprintf('\n🔍 Running quality checks before tests...\n');
    try
        quality_results = matlab_quality_config();
        results.quality_checks = quality_results;

        if quality_results.passed
            fprintf('✅ Quality checks passed\n');
        else
            fprintf('⚠️  Quality checks failed - continuing with tests\n');
        end
    catch ME
        fprintf('⚠️  Quality checks failed: %s\n', ME.message);
        results.quality_checks.error = ME.message;
    end

    % 3. Discover and run test files
    fprintf('\n🧪 Discovering test files...\n');

    % Find all test files
    test_dir = fullfile(pwd, 'tests');
    if ~exist(test_dir, 'dir')
        fprintf('❌ Test directory not found: %s\n', test_dir);
        results.summary = '❌ Test directory not found';
        return;
    end

    % Look for test files
    test_files = dir(fullfile(test_dir, 'test_*.m'));
    if isempty(test_files)
        fprintf('⚠️  No test files found in %s\n', test_dir);
        results.summary = '⚠️  No test files found';
        return;
    end

    fprintf('Found %d test files:\n', length(test_files));
    for i = 1:length(test_files)
        fprintf('  - %s\n', test_files(i).name);
    end

    % 4. Run each test file
    fprintf('\n🧪 Running individual tests...\n');

    for i = 1:length(test_files)
        test_file = fullfile(test_dir, test_files(i).name);
        fprintf('\nRunning: %s\n', test_files(i).name);

        try
            % Run the test function
            test_function = str2func(test_files(i).name(1:end-2)); % Remove .m extension
            test_results = test_function();

            if isstruct(test_results) && isfield(test_results, 'passed')
                if test_results.passed
                    fprintf('  ✅ PASSED\n');
                    results.tests_passed = results.tests_passed + 1;
                else
                    fprintf('  ❌ FAILED\n');
                    results.tests_failed = results.tests_failed + 1;
                    if isfield(test_results, 'issues')
                        for j = 1:length(test_results.issues)
                            fprintf('    Issue: %s\n', test_results.issues{j});
                        end
                    end
                end
            else
                fprintf('  ⚠️  UNKNOWN RESULT\n');
            end

            results.tests_run = results.tests_run + 1;

        catch ME
            fprintf('  ❌ ERROR: %s\n', ME.message);
            results.tests_failed = results.tests_failed + 1;
            results.errors{end+1} = sprintf('%s: %s', test_files(i).name, ME.message);
        end
    end

    % 5. Run MATLAB Unit Test Framework tests (if available)
    fprintf('\n🧪 Running MATLAB Unit Test Framework tests...\n');

    try
        % Create test suite from test directory
        suite = matlab.unittest.TestSuite.fromFolder(test_dir);

        if ~isempty(suite)
            fprintf('Found %d test methods in framework\n', length(suite));

            % Create test runner
            runner = matlab.unittest.TestRunner.withPlugin(...
                matlab.unittest.plugins.VerbosityPlugin(2)); % Verbose output

            % Run tests
            test_results = runner.run(suite);

            % Process results
            framework_tests_run = length(test_results);
            framework_tests_passed = sum([test_results.Passed]);
            framework_tests_failed = framework_tests_run - framework_tests_passed;

            fprintf('Framework Results: %d passed, %d failed\n', ...
                framework_tests_passed, framework_tests_failed);

            % Add framework results to overall results
            results.framework_tests = struct();
            results.framework_tests.run = framework_tests_run;
            results.framework_tests.passed = framework_tests_passed;
            results.framework_tests.failed = framework_tests_failed;

        else
            fprintf('No framework tests found\n');
        end

    catch ME
        fprintf('⚠️  Framework tests failed: %s\n', ME.message);
        results.errors{end+1} = sprintf('Framework tests: %s', ME.message);
    end

    % 6. Generate summary
    total_passed = results.tests_passed;
    total_failed = results.tests_failed;

    if isfield(results, 'framework_tests')
        total_passed = total_passed + results.framework_tests.passed;
        total_failed = total_failed + results.framework_tests.failed;
    end

    if total_failed == 0
        results.summary = sprintf('✅ All MATLAB tests PASSED (%d tests run)', results.tests_run);
        fprintf('\n%s\n', results.summary);
    else
        results.summary = sprintf('❌ MATLAB tests FAILED (%d passed, %d failed)', ...
            total_passed, total_failed);
        fprintf('\n%s\n', results.summary);
    end

    % 7. Save results to file
    try
        output_dir = fullfile(pwd, 'output', char(datetime('now', 'Format', 'yyyy-MM-dd')));
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        output_file = fullfile(output_dir, 'matlab_tests', 'test_results.json');
        output_dir_matlab = fullfile(output_dir, 'matlab_tests');
        if ~exist(output_dir_matlab, 'dir')
            mkdir(output_dir_matlab);
        end

        % Convert results to JSON-compatible format
        json_results = struct();
        json_results.timestamp = char(results.timestamp);
        json_results.tests_run = results.tests_run;
        json_results.tests_passed = results.tests_passed;
        json_results.tests_failed = results.tests_failed;
        json_results.summary = results.summary;

        if isfield(results, 'framework_tests')
            json_results.framework_tests = results.framework_tests;
        end

        % Write JSON file
        fid = fopen(output_file, 'w');
        if fid ~= -1
            fprintf(fid, '%s', jsonencode(json_results, 'PrettyPrint', true));
            fclose(fid);
            fprintf('Results saved to: %s\n', output_file);
        end

    catch ME
        fprintf('⚠️  Could not save results: %s\n', ME.message);
    end

    fprintf('\n✅ MATLAB test suite complete\n');
end
