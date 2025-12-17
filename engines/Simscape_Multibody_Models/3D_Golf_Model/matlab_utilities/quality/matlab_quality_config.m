function results = matlab_quality_config()
    % MATLAB_CODE_QUALITY_CONFIG Comprehensive quality checks for MATLAB code
    %
    % This function performs comprehensive quality checks on MATLAB code
    % following the project's .cursorrules.md requirements.
    %
    % Outputs:
    %   results - Struct containing quality check results
    %            .total_files - Total MATLAB files checked
    %            .issues - Array of quality issues found
    %            .passed - Boolean indicating if all checks passed
    %            .summary - Summary string of results

    fprintf('🔍 Setting up comprehensive MATLAB code quality checks...\n');

    % Initialize results structure
    results = struct();
    results.total_files = 0;
    results.issues = {};
    results.passed = true;
    results.timestamp = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss');

    % 1. Add current directory to path if not already there
    current_dir = pwd;
    path_modified = false;
    if ~contains(path, current_dir)
        addpath(current_dir);
        path_modified = true;
        fprintf('Added %s to MATLAB path\n', current_dir);
    end

    % Ensure path is cleaned up on function exit
    if path_modified
        cleanup_path = onCleanup(@() rmpath(current_dir));
    end

    % 2. Enable Code Analyzer warnings
    warning('on', 'all');

    % 3. Check for MATLAB Unit Testing Framework
    if exist('matlab.unittest.TestRunner', 'class')
        fprintf('✅ MATLAB Unit Testing Framework available\n');
        results.testing_framework = true;
    else
        fprintf('⚠️  MATLAB Unit Testing Framework not available\n');
        results.testing_framework = false;
        results.passed = false;
        results.issues{end+1} = 'MATLAB Unit Testing Framework not available';
    end

    % 4. Find all MATLAB files
    fprintf('\n📋 Scanning MATLAB files...\n');
    all_files = dir('**/*.m');

    % Filter files
    m_files = [];
    % Exclude folders containing these strings
    excluded_dirs = {'Archive', 'Backup', 'Old Revs', 'Older Revs', 'Python Version', 'legacy', 'examples', 'demos'};
    % Exclude files starting with these strings
    excluded_prefixes = {'Copy_of_', 'Safe_Copy_of_'};

    excluded_count = 0;

    for i = 1:length(all_files)
        file_path = fullfile(all_files(i).folder, all_files(i).name);

        % Check directory exclusions
        is_excluded = false;
        for j = 1:length(excluded_dirs)
            if contains(file_path, excluded_dirs{j}, 'IgnoreCase', true)
                is_excluded = true;
                break;
            end
        end

        if is_excluded
            excluded_count = excluded_count + 1;
            continue;
        end

        % Check filename exclusions
        for j = 1:length(excluded_prefixes)
            if startsWith(all_files(i).name, excluded_prefixes{j}, 'IgnoreCase', true)
                is_excluded = true;
                break;
            end
        end

        if is_excluded
            excluded_count = excluded_count + 1;
            continue;
        end

        if isempty(m_files)
            m_files = all_files(i);
        else
            m_files(end+1) = all_files(i);
        end
    end

    results.total_files = length(m_files);
    fprintf('Found %d active MATLAB files (filtered %d files)\n', results.total_files, excluded_count);

    % 5. Run comprehensive quality checks
    fprintf('\n🔍 Running comprehensive quality checks...\n');

    % Check each file for quality issues
    for i = 1:length(m_files)
        file_path = fullfile(m_files(i).folder, m_files(i).name);
        fprintf('Checking: %s\n', file_path);

        % Run checkcode on individual file
        try
            % Use checkcode instead of mlint
            file_issues = checkcode(file_path);
            if ~isempty(file_issues)
                for j = 1:length(file_issues)
                    issue_msg = sprintf('%s (line %d): %s', ...
                        m_files(i).name, file_issues(j).line, file_issues(j).message);
                    results.issues{end+1} = issue_msg;
                    results.passed = false;
                end
            end
        catch ME
            issue_msg = sprintf('%s: checkcode failed - %s', m_files(i).name, ME.message);
            results.issues{end+1} = issue_msg;
            results.passed = false;
        end

        % Check for required function structure (if it's a function file)
        if ~isempty(m_files(i).name) && ~strcmp(m_files(i).name(1), '%')
            try
                file_issues = check_function_structure(file_path);
                if ~isempty(file_issues)
                    results.issues = [results.issues, file_issues];
                    results.passed = false;
                end
            catch ME
                % Skip structure checks for files that can't be parsed
                fprintf('  ⚠️  Could not check structure: %s\n', ME.message);
            end
        end
    end

    % 7. Generate summary
    if results.passed
        results.summary = sprintf('✅ All MATLAB quality checks PASSED (%d files checked)', results.total_files);
        fprintf('\n%s\n', results.summary);
    else
        results.summary = sprintf('❌ MATLAB quality checks FAILED (%d files checked, %d issues found)', ...
            results.total_files, length(results.issues));
        fprintf('\n%s\n', results.summary);
        fprintf('\nIssues found:\n');
        for i = 1:length(results.issues)
            fprintf('  %d. %s\n', i, results.issues{i});
        end
    end

    fprintf('\n✅ MATLAB quality configuration complete\n');
end

function issues = check_function_structure(file_path)
    % CHECK_FUNCTION_STRUCTURE Check if function follows required structure
    %
    % This function checks if a MATLAB function follows the required
    % structure from .cursorrules.md

    arguments
        file_path (1,1) string
    end

    issues = {};

    try
        % Read file content
        fid = fopen(file_path, 'r');
        if fid == -1
            return;
        end

        content = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
        fclose(fid);

        if isempty(content{1})
            return;
        end

        lines = content{1};

        % Check for function definition
        has_function = false;
        has_docstring = false;
        has_arguments = false;

        for i = 1:length(lines)
            line = strtrim(lines{i});

            % Skip empty lines and comments
            if isempty(line) || startsWith(line, '%')
                continue;
            end

            % Check for function definition
            if startsWith(line, 'function')
                has_function = true;

                % Check if next non-empty line has docstring
                for j = i+1:length(lines)
                    next_line = strtrim(lines{j});
                    if ~isempty(next_line)
                        if startsWith(next_line, '%')
                            has_docstring = true;
                        end
                        break;
                    end
                end

                % Check for arguments validation
                for j = i+1:length(lines)
                    next_line = strtrim(lines{j});
                    if ~isempty(next_line) && ~startsWith(next_line, '%')
                        if contains(next_line, 'arguments')
                            has_arguments = true;
                        end
                        break;
                    end
                end

                break;
            end
        end

        % Generate issues for missing requirements
        if has_function && ~has_docstring
            issues{end+1} = sprintf('%s: Missing function docstring', file_path);
        end

        if has_function && ~has_arguments
            issues{end+1} = sprintf('%s: Missing arguments validation block', file_path);
        end

    catch ME
        % If we can't parse the file, skip structure checks
        issues{end+1} = sprintf('%s: Could not parse file structure - %s', file_path, ME.message);
    end
end
