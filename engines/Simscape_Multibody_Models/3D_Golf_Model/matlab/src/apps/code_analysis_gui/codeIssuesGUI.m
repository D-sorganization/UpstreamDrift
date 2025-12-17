function T = codeIssuesGUI(varargin)
%CODEISSUESGUI Interactive GUI for MATLAB Code Analyzer with file/folder selection.
%
%   T = CODEISSUESGUI() opens a GUI allowing users to select files or
%   folders for code analysis using MATLAB's Code Analyzer (MLint).
%
%   T = CODEISSUESGUI('Name', Value, ...) accepts these options:
%       'DefaultPath'   - Initial directory to show in file browser
%       'Output'        - Output file path. If omitted, user will be prompted.
%       'AutoSave'      - true|false (default false). If true, automatically
%                         saves results to a timestamped file.
%       'Recursive'     - true|false (default true). Only relevant if
%                         selecting folders.
%       'IncludeExt'    - Cellstr of file extensions to include. Default {'.m'}
%       'ExcludeDirs'   - Cellstr of directory names to skip
%       'ExcludeFiles'  - Cellstr of wildcard file patterns to skip
%       'OnError'       - 'record' (default) | 'rethrow'
%       'ShowProgress'  - true|false (default true). Shows progress dialog.
%
%   The returned table has the same format as exportCodeIssues:
%       File        - Absolute file path
%       RelFile     - Relative file path
%       Line        - Line number (double)
%       Column      - Column number (double)
%       Identifier  - Code Analyzer message identifier (string)
%       Message     - Human-readable message text (string)
%
%   Example:
%       % Open GUI for interactive code analysis
%       issues = codeIssuesGUI();
%
%       % Start in a specific directory
%       issues = codeIssuesGUI('DefaultPath', 'C:\MyProject\src');
%
%   See also: EXPORTCODEISSUES, CHECKCODE

% Parse inputs
p = inputParser;
p.addParameter('DefaultPath', pwd, @(s)ischar(s) || isstring(s));
p.addParameter('Output', '', @(s)ischar(s) || isstring(s));
p.addParameter('AutoSave', false, @(x)islogical(x) && isscalar(x));
p.addParameter('Recursive', true, @(x)islogical(x) && isscalar(x));
p.addParameter('IncludeExt', {'.m'}, @(c)iscellstr(c) || isstring(c));
p.addParameter('ExcludeDirs', {}, @(c)iscellstr(c) || isstring(c));
p.addParameter('ExcludeFiles', {}, @(c)iscellstr(c) || isstring(c));
p.addParameter('OnError', 'record', @(s)any(strcmpi(s,{'record','rethrow'})));
p.addParameter('ShowProgress', true, @(x)islogical(x) && isscalar(x));
p.parse(varargin{:});
opts = p.Results;

% Normalize paths
opts.DefaultPath = char(string(opts.DefaultPath));
opts.Output = char(string(opts.Output));

% Show selection dialog
[selectedPaths, analysisOpts] = showSelectionDialog(opts);

if isempty(selectedPaths)
    % User cancelled
    T = table();
    return;
end

% Process each selected path
T = table();
for i = 1:length(selectedPaths)
    targetPath = selectedPaths{i};

    if opts.ShowProgress
        if length(selectedPaths) > 1
            progressMsg = sprintf('Analyzing %d of %d: %s', i, length(selectedPaths), targetPath);
        else
            progressMsg = sprintf('Analyzing: %s', targetPath);
        end
        fprintf('[codeIssuesGUI] %s\n', progressMsg);
    end

    % Call the core exportCodeIssues function
    try
        pathResults = exportCodeIssues(targetPath, ...
            'Recursive', analysisOpts.Recursive, ...
            'IncludeExt', analysisOpts.IncludeExt, ...
            'ExcludeDirs', analysisOpts.ExcludeDirs, ...
            'ExcludeFiles', analysisOpts.ExcludeFiles, ...
            'OnError', analysisOpts.OnError, ...
            'Quiet', ~opts.ShowProgress);

        T = [T; pathResults]; %#ok<AGROW>

    catch ME
        if strcmpi(analysisOpts.OnError, 'rethrow')
            rethrow(ME);
        else
            fprintf('[codeIssuesGUI] Error analyzing %s: %s\n', targetPath, ME.message);
        end
    end
end

% Sort results
if height(T) > 0
    T = sortrows(T, {'RelFile','Line','Column','Identifier'});
end

% Handle output
if ~isempty(analysisOpts.OutputFile) || opts.AutoSave
    outputFile = analysisOpts.OutputFile;
    if isempty(outputFile) && opts.AutoSave
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        outputFile = fullfile(pwd, sprintf('code_issues_%s.csv', timestamp));
    end

    if ~isempty(outputFile)
        try
            exportCodeIssues('dummy', 'Output', ''); % Just to use the writeOutput function
            % Actually, let's implement our own simple output
            writetable(T, outputFile);
            fprintf('[codeIssuesGUI] Results saved to: %s\n', outputFile);
        catch ME
            fprintf('[codeIssuesGUI] Warning: Could not save to %s: %s\n', outputFile, ME.message);
        end
    end
end

% Show summary
if opts.ShowProgress
    if height(T) > 0
        fprintf('[codeIssuesGUI] Analysis complete: %d issues found\n', height(T));
    else
        fprintf('[codeIssuesGUI] Analysis complete: No issues found\n');
    end
end

end

function [selectedPaths, analysisOpts] = showSelectionDialog(opts)
%SHOWSELECTIONDIALOG Show dialog for path and option selection

selectedPaths = {};
analysisOpts = struct();

% Create main dialog figure
fig = figure('Name', 'MATLAB Code Issues Analyzer', ...
    'NumberTitle', 'off', ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    'Resize', 'off', ...
    'Position', [100, 100, 500, 400], ...
    'WindowStyle', 'modal');

% Main layout
mainPanel = uipanel('Parent', fig, 'Position', [0.02 0.02 0.96 0.96], ...
    'Title', 'Select Files/Folders for Analysis');

% File/folder selection section
selectionPanel = uipanel('Parent', mainPanel, 'Position', [0.02 0.65 0.96 0.33], ...
    'Title', 'Target Selection');

% Selected paths listbox
pathListLabel = uicontrol('Parent', selectionPanel, 'Style', 'text', ...
    'String', 'Selected paths:', 'Position', [10 80 100 20], ...
    'HorizontalAlignment', 'left');

pathListbox = uicontrol('Parent', selectionPanel, 'Style', 'listbox', ...
    'Position', [10 20 350 60], 'Max', 10);

% Buttons for adding/removing paths
addFileBtn = uicontrol('Parent', selectionPanel, 'Style', 'pushbutton', ...
    'String', 'Add File(s)', 'Position', [370 80 100 25], ...
    'Callback', @addFiles);

addFolderBtn = uicontrol('Parent', selectionPanel, 'Style', 'pushbutton', ...
    'String', 'Add Folder', 'Position', [370 50 100 25], ...
    'Callback', @addFolder);

removeBtn = uicontrol('Parent', selectionPanel, 'Style', 'pushbutton', ...
    'String', 'Remove', 'Position', [370 20 100 25], ...
    'Callback', @removePath);

% Analysis options section
optionsPanel = uipanel('Parent', mainPanel, 'Position', [0.02 0.25 0.96 0.38], ...
    'Title', 'Analysis Options');

% Recursive checkbox
recursiveCheck = uicontrol('Parent', optionsPanel, 'Style', 'checkbox', ...
    'String', 'Recursive (scan subfolders)', 'Value', opts.Recursive, ...
    'Position', [10 100 200 20]);

% File extensions
extLabel = uicontrol('Parent', optionsPanel, 'Style', 'text', ...
    'String', 'File extensions:', 'Position', [10 75 100 20], ...
    'HorizontalAlignment', 'left');

extEdit = uicontrol('Parent', optionsPanel, 'Style', 'edit', ...
    'String', strjoin(opts.IncludeExt, ', '), 'Position', [110 75 150 20]);

% Exclude directories
excludeDirLabel = uicontrol('Parent', optionsPanel, 'Style', 'text', ...
    'String', 'Exclude dirs:', 'Position', [10 50 100 20], ...
    'HorizontalAlignment', 'left');

excludeDirEdit = uicontrol('Parent', optionsPanel, 'Style', 'edit', ...
    'String', strjoin(opts.ExcludeDirs, ', '), 'Position', [110 50 200 20]);

% Exclude file patterns
excludeFileLabel = uicontrol('Parent', optionsPanel, 'Style', 'text', ...
    'String', 'Exclude files:', 'Position', [10 25 100 20], ...
    'HorizontalAlignment', 'left');

excludeFileEdit = uicontrol('Parent', optionsPanel, 'Style', 'edit', ...
    'String', strjoin(opts.ExcludeFiles, ', '), 'Position', [110 25 200 20]);

% Error handling
errorPopup = uicontrol('Parent', optionsPanel, 'Style', 'popupmenu', ...
    'String', {'Record errors as issues', 'Stop on first error'}, ...
    'Value', 1, 'Position', [320 50 150 20]);

% Output section
outputPanel = uipanel('Parent', mainPanel, 'Position', [0.02 0.05 0.96 0.18], ...
    'Title', 'Output Options');

outputEdit = uicontrol('Parent', outputPanel, 'Style', 'edit', ...
    'String', opts.Output, 'Position', [10 20 300 20]);

browseOutputBtn = uicontrol('Parent', outputPanel, 'Style', 'pushbutton', ...
    'String', 'Browse...', 'Position', [320 20 80 20], ...
    'Callback', @browseOutput);

autoSaveCheck = uicontrol('Parent', outputPanel, 'Style', 'checkbox', ...
    'String', 'Auto-save with timestamp', 'Value', opts.AutoSave, ...
    'Position', [410 20 150 20]);

% Control buttons
analyzeBtn = uicontrol('Parent', mainPanel, 'Style', 'pushbutton', ...
    'String', 'Analyze', 'Position', [300 10 80 30], ...
    'FontWeight', 'bold', 'Callback', @analyzeCode);

cancelBtn = uicontrol('Parent', mainPanel, 'Style', 'pushbutton', ...
    'String', 'Cancel', 'Position', [390 10 80 30], ...
    'Callback', @cancelDialog);

% Store current paths
currentPaths = {};

    function addFiles(~, ~)
        [files, path] = uigetfile({'*.m', 'MATLAB Files (*.m)'; ...
            '*.*', 'All Files (*.*)'}, ...
            'Select MATLAB files', opts.DefaultPath, 'MultiSelect', 'on');

        if ~isequal(files, 0)
            if ischar(files)
                files = {files};
            end
            for i = 1:length(files)
                newPath = fullfile(path, files{i});
                if ~any(strcmp(currentPaths, newPath))
                    currentPaths{end+1} = newPath;
                end
            end
            updatePathList();
        end
    end

    function addFolder(~, ~)
        folder = uigetdir(opts.DefaultPath, 'Select folder to analyze');
        if ~isequal(folder, 0)
            if ~any(strcmp(currentPaths, folder))
                currentPaths{end+1} = folder;
            end
            updatePathList();
        end
    end

    function removePath(~, ~)
        selected = get(pathListbox, 'Value');
        if ~isempty(selected) && selected <= length(currentPaths)
            currentPaths(selected) = [];
            updatePathList();
        end
    end

    function updatePathList()
        set(pathListbox, 'String', currentPaths, 'Value', min(get(pathListbox, 'Value'), length(currentPaths)));
    end

    function browseOutput(~, ~)
        [file, path] = uiputfile({'*.csv', 'CSV Files (*.csv)'; ...
            '*.xlsx', 'Excel Files (*.xlsx)'; ...
            '*.json', 'JSON Files (*.json)'; ...
            '*.md', 'Markdown Files (*.md)'}, ...
            'Save analysis results', 'code_issues.csv');

        if ~isequal(file, 0)
            set(outputEdit, 'String', fullfile(path, file));
        end
    end

    function analyzeCode(~, ~)
        if isempty(currentPaths)
            uiwait(msgbox('Please select at least one file or folder to analyze.', 'No Selection', 'warn'));
            return;
        end

        % Collect options
        selectedPaths = currentPaths;

        analysisOpts.Recursive = get(recursiveCheck, 'Value');

        extStr = get(extEdit, 'String');
        if isempty(strtrim(extStr))
            analysisOpts.IncludeExt = {'.m'};
        else
            analysisOpts.IncludeExt = strtrim(strsplit(extStr, ','));
        end

        excludeDirStr = get(excludeDirEdit, 'String');
        if isempty(strtrim(excludeDirStr))
            analysisOpts.ExcludeDirs = {};
        else
            analysisOpts.ExcludeDirs = strtrim(strsplit(excludeDirStr, ','));
        end

        excludeFileStr = get(excludeFileEdit, 'String');
        if isempty(strtrim(excludeFileStr))
            analysisOpts.ExcludeFiles = {};
        else
            analysisOpts.ExcludeFiles = strtrim(strsplit(excludeFileStr, ','));
        end

        errorOpts = {'record', 'rethrow'};
        analysisOpts.OnError = errorOpts{get(errorPopup, 'Value')};

        analysisOpts.OutputFile = get(outputEdit, 'String');

        close(fig);
    end

    function cancelDialog(~, ~)
        selectedPaths = {};
        analysisOpts = struct();
        close(fig);
    end

% Wait for dialog to close
uiwait(fig);

end
