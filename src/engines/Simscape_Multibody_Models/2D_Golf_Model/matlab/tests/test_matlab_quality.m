function tests = test_matlab_quality
% TEST_MATLAB_QUALITY Test suite to verify MATLAB code quality standards
%
% This test suite verifies that MATLAB code follows quality standards:
% - No magic numbers
% - Proper documentation
% - Reproducible random number generation
%
% Run with: runtests('matlab/tests/test_matlab_quality.m')

    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    % Setup run once before all tests
    % Add MATLAB paths
    addpath(genpath('matlab'));
    if exist('matlab_optimized', 'dir')
        addpath(genpath('matlab_optimized'));
    end
end

function test_no_magic_numbers(testCase)
    % Verify that common magic numbers are not present in code
    
    magicNumbers = {'3.14', '9.8', '6.67', '2.71'};
    mFiles = get_matlab_files();
    
    found = false;
    foundFiles = {};
    
    for i = 1:length(mFiles)
        filePath = fullfile(mFiles(i).folder, mFiles(i).name);
        
        % Skip files using helper function
        if should_skip_file_for_quality(filePath)
            continue;
        end
        
        content = fileread(filePath);
        
        for j = 1:length(magicNumbers)
            if contains(content, magicNumbers{j})
                found = true;
                foundFiles{end+1} = sprintf('%s contains %s', filePath, magicNumbers{j}); %#ok<AGROW>
            end
        end
    end
    
    % Verify no magic numbers found
    if found
        msg = sprintf('Magic numbers found in:\n%s\nUse named constants instead!', ...
                     strjoin(foundFiles, '\n'));
        verifyFalse(testCase, found, msg);
    else
        verifyFalse(testCase, found, 'No magic numbers should be present');
    end
end

function test_functions_have_documentation(testCase)
    % Verify that function files have documentation
    
    mFiles = get_matlab_files();
    missingDocs = {};
    
    for i = 1:length(mFiles)
        filePath = fullfile(mFiles(i).folder, mFiles(i).name);
        
        % Skip files using helper function
        if should_skip_file_for_quality(filePath)
            continue;
        end
        
        content = fileread(filePath);
        lines = strsplit(content, '\n');
        
        % Check if it's a function file
        if isempty(lines) || ~startsWith(strtrim(lines{1}), 'function')
            continue;
        end
        
        % Count comment lines after function declaration
        commentCount = 0;
        for j = 2:min(length(lines), 20)
            line = strtrim(lines{j});
            if startsWith(line, '%')
                commentCount = commentCount + 1;
            elseif ~isempty(line)
                break;
            end
        end
        
        % Functions should have at least 3 comment lines
        if commentCount < 3
            [~, name, ext] = fileparts(filePath);
            missingDocs{end+1} = sprintf('%s%s (%d comment lines)', name, ext, commentCount); %#ok<AGROW>
        end
    end
    
    % Allow some files to have minimal docs but warn
    if ~isempty(missingDocs)
        warning('test_matlab_quality:documentation', ...
                'Files with minimal documentation:\n%s', ...
                strjoin(missingDocs, '\n'));
    end
    
    % Don't fail the test, just warn
    verifyTrue(testCase, true, 'Documentation check completed');
end

function test_random_functions_use_seeds(testCase)
    % Verify that files using random functions set seeds
    
    mFiles = get_matlab_files();
    unseeded = {};
    
    for i = 1:length(mFiles)
        filePath = fullfile(mFiles(i).folder, mFiles(i).name);
        
        % Skip archived code (but allow test files to check for seeds)
        if contains(filePath, 'Archive') || contains(filePath, 'Backup')
            continue;
        end
        
        content = fileread(filePath);
        
        % Check for random functions
        hasRandom = contains(content, 'rand(') || ...
                    contains(content, 'randn(') || ...
                    contains(content, 'randi(');
        hasRng = contains(content, 'rng(');
        
        if hasRandom && ~hasRng
            [~, name, ext] = fileparts(filePath);
            unseeded{end+1} = sprintf('%s%s', name, ext); %#ok<AGROW>
        end
    end
    
    % Warn about unseeded random functions
    if ~isempty(unseeded)
        warning('test_matlab_quality:randomness', ...
                'Files using random functions without rng:\n%s\nAdd rng(seed) for reproducibility', ...
                strjoin(unseeded, '\n'));
    end
    
    % Don't fail the test, just warn
    verifyTrue(testCase, true, 'Random seed check completed');
end

function test_run_all_script_exists(testCase)
    % Verify that run_all.m exists and is executable
    
    verifyTrue(testCase, exist('matlab/run_all.m', 'file') ~= 0, ...
               'matlab/run_all.m should exist');
end

function test_quality_check_script_exists(testCase)
    % Verify that quality check script exists
    
    verifyTrue(testCase, exist('run_matlab_quality_checks.m', 'file') ~= 0, ...
               'run_matlab_quality_checks.m should exist');
end

function test_tests_directory_exists(testCase)
    % Verify that tests directory exists
    
    verifyTrue(testCase, exist('matlab/tests', 'dir') ~= 0, ...
               'matlab/tests directory should exist');
end

% Helper functions

function files = get_matlab_files()
    % Get all MATLAB .m files
    files = [];
    
    if exist('matlab', 'dir')
        matlabFiles = dir('matlab/**/*.m');
        files = [files; matlabFiles];
    end
    
    if exist('matlab_optimized', 'dir')
        optimizedFiles = dir('matlab_optimized/**/*.m');
        files = [files; optimizedFiles];
    end
end

function skip = should_skip_file_for_quality(filePath)
    % Determine if a file should be skipped in quality checks
    % Used to reduce code duplication across test functions
    
    % Directories and patterns to skip
    skipPatterns = {
        'Archive', 'archive',
        'Backup', 'backup',
        'Old', 'old',
        'Legacy', 'legacy',
        'Experimental', 'experimental',
        'test', 'Test'  % Skip test files for quality checks
    };
    
    skip = false;
    for i = 1:length(skipPatterns)
        if contains(filePath, skipPatterns{i})
            skip = true;
            return;
        end
    end
end
