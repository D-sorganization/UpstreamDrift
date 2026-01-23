function results = run_quality_checks(target_path, varargin)
    % RUN_QUALITY_CHECKS Unified MATLAB quality checking interface
    %
    % This function provides a unified interface for running comprehensive
    % quality checks on MATLAB code, combining mlint analysis with custom
    % quality rules.
    %
    % Inputs:
    %   target_path - Path to file or directory to analyze (optional)
    %                 If omitted, analyzes current directory
    %
    % Name-Value Arguments:
    %   'OutputFile'    - Path to save results (CSV, XLSX, JSON, or MD)
    %   'Recursive'     - true|false (default: true) scan subdirectories
    %   'ExcludeDirs'   - Cell array of directory names to exclude
    %   'Verbose'       - true|false (default: true) print progress
    %   'StrictMode'    - true|false (default: false) fail on any issues
    %
    % Outputs:
    %   results - Struct containing:
    %             .issues_table  - Table of all issues found
    %             .total_files   - Number of files checked
    %             .total_issues  - Number of issues found
    %             .passed        - Boolean indicating if all checks passed
    %             .summary       - Summary message
    %             .timestamp     - When analysis was run
    %
    % Examples:
    %   % Check current directory
    %   results = run_quality_checks();
    %
    %   % Check specific directory and save to CSV
    %   results = run_quality_checks('../src', 'OutputFile', 'quality_report.csv');
    %
    %   % Strict mode (fails on any issues)
    %   results = run_quality_checks('.', 'StrictMode', true);
    %
    % See also: exportCodeIssues, matlab_quality_config

    arguments
        target_path (1,1) string = pwd
    end

    % Parse additional arguments
    p = inputParser;
    p.addParameter('OutputFile', '', @(x) ischar(x) || isstring(x));
    p.addParameter('Recursive', true, @islogical);
    p.addParameter('ExcludeDirs', {'.git', 'slprj', 'codegen', 'build'}, @iscell);
    p.addParameter('Verbose', true, @islogical);
    p.addParameter('StrictMode', false, @islogical);
    p.parse(varargin{:});
    opts = p.Results;

    % Initialize results
    results = struct();
    results.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');
    results.target_path = char(target_path);

    if opts.Verbose
        fprintf('\n========================================\n');
        fprintf('MATLAB Quality Checks\n');
        fprintf('========================================\n');
        fprintf('Target: %s\n', target_path);
        fprintf('Timestamp: %s\n', results.timestamp);
        fprintf('========================================\n\n');
    end

    % Run exportCodeIssues for comprehensive analysis
    try
        if opts.Verbose
            fprintf('Running Code Analyzer (mlint)...\n');
        end

        issues_table = exportCodeIssues(target_path, ...
            'Recursive', opts.Recursive, ...
            'ExcludeDirs', opts.ExcludeDirs, ...
            'Quiet', ~opts.Verbose);

        results.issues_table = issues_table;
        results.total_issues = height(issues_table);

    catch ME
        fprintf('ERROR: Failed to run code analysis: %s\n', ME.message);
        results.issues_table = table();
        results.total_issues = -1;
        results.passed = false;
        results.summary = sprintf('FAILED - Error during analysis: %s', ME.message);
        return;
    end

    % Count files analyzed
    if ~isempty(results.issues_table)
        results.total_files = numel(unique(results.issues_table.File));
    else
        % Count .m files in target path
        if isfolder(target_path)
            if opts.Recursive
                files = dir(fullfile(target_path, '**', '*.m'));
            else
                files = dir(fullfile(target_path, '*.m'));
            end
            results.total_files = length(files);
        else
            results.total_files = 1;
        end
    end

    % Determine pass/fail
    if opts.StrictMode
        results.passed = (results.total_issues == 0);
    else
        % In non-strict mode, only fail on critical issues
        % (For now, we pass as long as analysis completed)
        results.passed = (results.total_issues >= 0);
    end

    % Generate summary
    if results.total_issues == 0
        results.summary = sprintf('✅ PASSED - No issues found (%d files checked)', ...
            results.total_files);
    elseif results.passed
        results.summary = sprintf('⚠️  PASSED (with warnings) - %d issues found in %d files', ...
            results.total_issues, results.total_files);
    else
        results.summary = sprintf('❌ FAILED - %d issues found in %d files', ...
            results.total_issues, results.total_files);
    end

    % Print summary
    if opts.Verbose
        fprintf('\n========================================\n');
        fprintf('%s\n', results.summary);
        fprintf('========================================\n\n');

        if results.total_issues > 0 && results.total_issues <= 20
            fprintf('Issues found:\n');
            for i = 1:height(results.issues_table)
                fprintf('  %d. %s (line %d): %s\n', ...
                    i, ...
                    results.issues_table.RelFile{i}, ...
                    results.issues_table.Line(i), ...
                    results.issues_table.Message{i});
            end
        elseif results.total_issues > 20
            fprintf('First 20 issues:\n');
            for i = 1:20
                fprintf('  %d. %s (line %d): %s\n', ...
                    i, ...
                    results.issues_table.RelFile{i}, ...
                    results.issues_table.Line(i), ...
                    results.issues_table.Message{i});
            end
            fprintf('  ... and %d more issues\n', results.total_issues - 20);
        end
    end

    % Save results if requested
    if ~isempty(opts.OutputFile)
        try
            % exportCodeIssues already handles different formats
            writetable(results.issues_table, opts.OutputFile);
            if opts.Verbose
                fprintf('\nResults saved to: %s\n', opts.OutputFile);
            end
        catch ME
            fprintf('WARNING: Could not save results: %s\n', ME.message);
        end
    end

end
