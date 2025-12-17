function run_matlab_quality_checks()
% RUN_MATLAB_QUALITY_CHECKS Run all MATLAB quality and compliance checks
%
% This function performs comprehensive quality checks on MATLAB code including:
% - Magic number detection
% - Code analyzer (checkcode)
% - Reproducibility checks (random seeds)
% - Documentation validation
%
% Usage:
%   run_matlab_quality_checks()
%
% Requirements:
%   MATLAB R2016b or later (for recursive dir pattern)
%
% The function will display results for each check and report any issues found.
%
% See also: checkcode, runtests

    fprintf('🔍 Running MATLAB Quality Checks...\n\n');
    
    % Track overall status
    allChecksPassed = true;
    
    try
        % 1. Check for magic numbers
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        magicNumbersOk = check_magic_numbers();
        allChecksPassed = allChecksPassed && magicNumbersOk;
        
        % 2. Run code analyzer
        fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        codeAnalyzerOk = run_code_analyzer();
        allChecksPassed = allChecksPassed && codeAnalyzerOk;
        
        % 3. Check reproducibility
        fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        reproducibilityOk = check_reproducibility();
        allChecksPassed = allChecksPassed && reproducibilityOk;
        
        % 4. Validate documentation
        fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        documentationOk = check_documentation();
        allChecksPassed = allChecksPassed && documentationOk;
        
        % Final summary
        fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        if allChecksPassed
            fprintf('✅ All MATLAB quality checks completed successfully!\n');
        else
            fprintf('⚠️  Some quality checks found issues (see above)\n');
        end
        fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        
    catch ME
        fprintf('\n❌ Error during quality checks:\n');
        fprintf('   %s\n', ME.message);
        fprintf('   in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        error('Quality checks failed with error');
    end
end

function ok = check_magic_numbers()
    % Check for magic numbers in MATLAB code
    fprintf('1️⃣  Checking for magic numbers...\n\n');
    
    % Common magic numbers to avoid (with suggested alternatives)
    % Use more precise patterns to avoid false positives
    magicNumbers = struct(...
        'pattern', {'\<3\.14[0-9]*\>', '\<9\.8[0-9]*\>', '\<6\.67[0-9]*\>', '\<2\.71[0-9]*\>'}, ...
        'suggestion', {'pi (built-in)', 'g with source', 'G with source', 'exp(1)'} ...
    );
    
    files = get_matlab_files();
    found = false;
    fileCount = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived, test, and backup files
        if should_skip_file(filePath)
            continue;
        end
        
        fileCount = fileCount + 1;
        content = fileread(filePath);
        
        for j = 1:length(magicNumbers)
            % Use regexp for more precise pattern matching
            if ~isempty(regexp(content, magicNumbers(j).pattern, 'once'))
                fprintf('   ⚠️  Found magic number matching %s in %s\n', magicNumbers(j).pattern, filePath);
                fprintf('       → Suggestion: Use %s\n', magicNumbers(j).suggestion);
                found = true;
            end
        end
    end
    
    fprintf('\n   Checked %d MATLAB files\n', fileCount);
    
    if ~found
        fprintf('   ✅ No magic numbers found\n');
        ok = true;
    else
        fprintf('   ⚠️  Magic numbers detected - use named constants with sources\n');
        ok = false;
    end
end

function ok = run_code_analyzer()
    % Run MATLAB code analyzer (checkcode) on all files
    fprintf('2️⃣  Running code analyzer (checkcode)...\n\n');
    
    files = get_matlab_files();
    hasIssues = false;
    fileCount = 0;
    issueCount = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived, test, and backup files
        if should_skip_file(filePath)
            continue;
        end
        
        fileCount = fileCount + 1;
        
        try
            issues = checkcode(filePath, '-id');
            
            if ~isempty(issues)
                fprintf('   ⚠️  Issues in %s:\n', filePath);
                for j = 1:length(issues)
                    fprintf('      Line %d: %s\n', issues(j).line, issues(j).message);
                    issueCount = issueCount + 1;
                end
                fprintf('\n');
                hasIssues = true;
            end
        catch ME
            fprintf('   ⚠️  Error checking %s: %s\n', filePath, ME.message);
            hasIssues = true;
        end
    end
    
    fprintf('   Checked %d MATLAB files\n', fileCount);
    
    if ~hasIssues
        fprintf('   ✅ All files pass code analyzer\n');
        ok = true;
    else
        fprintf('   ⚠️  Found %d issues - please review and fix\n', issueCount);
        ok = false;
    end
end

function ok = check_reproducibility()
    % Check that random number generation uses seeds
    fprintf('3️⃣  Checking reproducibility (random seeds)...\n\n');
    
    files = get_matlab_files();
    warnings = 0;
    fileCount = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived files
        if should_skip_file(filePath)
            continue;
        end
        
        fileCount = fileCount + 1;
        content = fileread(filePath);
        
        % Check for random functions without rng (use word boundaries for accuracy)
        hasRandom = ~isempty(regexp(content, '\<rand\(', 'once')) || ...
                    ~isempty(regexp(content, '\<randn\(', 'once')) || ...
                    ~isempty(regexp(content, '\<randi\(', 'once'));
        hasRng = ~isempty(regexp(content, '\<rng\(', 'once'));
        
        if hasRandom && ~hasRng
            % Extract filename for cleaner display
            [~, name, ext] = fileparts(filePath);
            fprintf('   ⚠️  %s%s uses randomness without rng seed\n', name, ext);
            fprintf('       File: %s\n', filePath);
            warnings = warnings + 1;
        end
    end
    
    fprintf('\n   Checked %d MATLAB files\n', fileCount);
    
    if warnings == 0
        fprintf('   ✅ All files with randomness have seeds or are intentionally random\n');
        ok = true;
    else
        fprintf('   ⚠️  %d files use randomness without visible rng seed\n', warnings);
        fprintf('       Add rng(seed) at the beginning of scripts using random functions\n');
        ok = false;
    end
end

function ok = check_documentation()
    % Check for function documentation
    fprintf('4️⃣  Checking function documentation...\n\n');
    
    files = get_matlab_files();
    missingDocs = 0;
    poorDocs = 0;
    fileCount = 0;
    functionCount = 0;
    
    for i = 1:length(files)
        filePath = fullfile(files(i).folder, files(i).name);
        
        % Skip archived and test files
        if should_skip_file(filePath)
            continue;
        end
        
        fileCount = fileCount + 1;
        content = fileread(filePath);
        lines = strsplit(content, '\n');
        
        % Check if it's a function file (look in first 10 lines for function keyword)
        isFunction = false;
        for lineIdx = 1:min(10, length(lines))
            line = strtrim(lines{lineIdx});
            if startsWith(line, 'function') && ~startsWith(line, '%')
                isFunction = true;
                break;
            end
        end
        
        if ~isFunction
            continue;
        end
        
        functionCount = functionCount + 1;
        
        % Count comment lines at the beginning (after function declaration)
        commentCount = 0;
        for j = 2:min(length(lines), 50)  % Check first 50 lines
            line = strtrim(lines{j});
            if startsWith(line, '%')
                commentCount = commentCount + 1;
            elseif ~isempty(line)
                break;  % Stop at first non-comment, non-empty line
            end
        end
        
        % Extract filename for display
        [~, name, ext] = fileparts(filePath);
        
        if commentCount == 0
            fprintf('   ⚠️  %s%s has no documentation\n', name, ext);
            missingDocs = missingDocs + 1;
        elseif commentCount < 5
            fprintf('   ⚠️  %s%s has minimal documentation (%d lines)\n', name, ext, commentCount);
            poorDocs = poorDocs + 1;
        end
    end
    
    fprintf('\n   Checked %d MATLAB files (%d functions)\n', fileCount, functionCount);
    
    if missingDocs == 0 && poorDocs == 0
        fprintf('   ✅ All functions have adequate documentation\n');
        ok = true;
    else
        if missingDocs > 0
            fprintf('   ⚠️  %d functions missing documentation\n', missingDocs);
        end
        if poorDocs > 0
            fprintf('   ⚠️  %d functions have minimal documentation\n', poorDocs);
        end
        fprintf('       Add comprehensive function headers with inputs, outputs, examples\n');
        ok = false;
    end
end

function files = get_matlab_files()
    % Get all MATLAB .m files in matlab/ and matlab_optimized/ directories
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

function skip = should_skip_file(filePath)
    % Determine if a file should be skipped in quality checks
    
    % Directories to skip
    skipDirs = {
        'Archive', 'archive',
        'Backup', 'backup',
        'Old', 'old',
        'Legacy', 'legacy',
        'Experimental', 'experimental',
        'test', 'Test', 'tests', 'Tests'
    };
    
    skip = false;
    for i = 1:length(skipDirs)
        if contains(filePath, skipDirs{i})
            skip = true;
            return;
        end
    end
end
