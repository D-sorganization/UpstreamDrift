function launchCodeAnalyzer()
%LAUNCHCODEANALYZER Quick launcher for the interactive Code Issues GUI.
%
%   LAUNCHCODEANALYZER() opens the interactive MATLAB Code Analyzer GUI
%   that allows users to select files or folders for analysis and configure
%   analysis options through a user-friendly interface.
%
%   This is a convenience function that calls codeIssuesGUI() with default
%   settings optimized for typical use cases.
%
%   Example:
%       % Launch the GUI
%       launchCodeAnalyzer();
%
%       % Or equivalently:
%       codeIssuesGUI();
%
%   See also: CODEISSUESGUI, EXPORTCODEISSUES

fprintf('Launching MATLAB Code Issues Analyzer...\n');

try
    % Launch the GUI with sensible defaults
    results = codeIssuesGUI('ShowProgress', true, 'AutoSave', false);

    if ~isempty(results) && height(results) > 0
        fprintf('\nAnalysis Results Summary:\n');
        fprintf('========================\n');
        fprintf('Total issues found: %d\n', height(results));

        % Show breakdown by file
        if height(results) > 0
            uniqueFiles = unique(results.RelFile);
            fprintf('Files analyzed: %d\n', length(uniqueFiles));

            % Show top issue types
            if height(results) > 0
                uniqueIds = unique(results.Identifier);
                fprintf('Issue types found: %d\n', length(uniqueIds));

                % Count issues by type
                [counts, ids] = countIssueTypes(results);
                fprintf('\nTop issue types:\n');
                for i = 1:min(5, length(ids))
                    fprintf('  %s: %d occurrences\n', ids{i}, counts(i));
                end
            end
        end

        fprintf('\nTip: Results are also returned as a table variable.\n');
        fprintf('     Use assignin(''base'', ''codeIssues'', results) to save to workspace.\n');

        % Optionally assign to base workspace
        try
            assignin('base', 'codeAnalysisResults', results);
            fprintf('Results saved to workspace variable: codeAnalysisResults\n');
        catch
            % Silent fail if workspace assignment doesn't work
        end

    else
        fprintf('No issues found or analysis was cancelled.\n');
    end

catch ME
    fprintf('Error during code analysis: %s\n', ME.message);
    rethrow(ME);
end

end

function [counts, ids] = countIssueTypes(results)
%COUNTISSUETYPES Count occurrences of each issue type

uniqueIds = unique(results.Identifier);
counts = zeros(size(uniqueIds));

for i = 1:length(uniqueIds)
    counts(i) = sum(strcmp(results.Identifier, uniqueIds{i}));
end

% Sort by count (descending)
[counts, sortIdx] = sort(counts, 'descend');
ids = uniqueIds(sortIdx);

end
