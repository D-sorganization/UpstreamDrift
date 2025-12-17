function T = exportCodeIssues(targetPath, varargin)
%EXPORTCODEISSUES Scan MATLAB code with Code Analyzer (MLint) and export findings.
%
%   T = EXPORTCODEISSUES(targetPath) runs Code Analyzer on a single file or on
%   all .m files under a folder (recursively) and returns a table of issues.
%
%   T = EXPORTCODEISSUES(targetPath, 'Name', Value, ...) accepts these options:
%       'Output'        - Output file path. If omitted, no file is written.
%                         The file type is inferred from the extension:
%                           .csv  -> CSV via writetable
%                           .xlsx -> Excel via writetable
%                           .json -> JSON (UTF-8)
%                           .md   -> Markdown table
%       'Recursive'     - true|false (default true). Only relevant if
%                         targetPath is a folder.
%       'IncludeExt'    - Cellstr of file extensions to include. Default {'.m'}
%       'ExcludeDirs'   - Cellstr of directory names to skip (no paths), e.g.,
%                         {'.git','build','external'}. Default common set.
%       'ExcludeFiles'  - Cellstr of wildcard file patterns to skip, e.g.,
%                         {'*_autogen.m','*_mex.m'}. Default {}.
%       'Root'          - Root folder for computing relative paths in output.
%                         Default: if targetPath is a folder, that folder;
%                         if it is a file, fileparts(targetPath).
%       'OnError'       - 'record' (default) | 'rethrow'. If 'record', any
%                         checkcode errors are captured as pseudo-issues.
%       'Quiet'         - true|false (default true). If false, prints progress.
%
%   The returned table columns:
%       File        - Absolute file path
%       RelFile     - Relative file path (based on 'Root')
%       Line        - Line number (double)
%       Column      - Column number (double)
%       Identifier  - Code Analyzer message identifier (string)
%       Message     - Human-readable message text (string)
%
%   Example:
%       % Single file -> CSV
%       exportCodeIssues('C:\MyProj\src\foo.m', 'Output','issues.csv');
%
%       % Whole folder (recursive) -> Excel
%       exportCodeIssues('C:\MyProj', 'Output','lint.xlsx');
%
%   Notes:
%   * Uses CHECKCODE under the hood with '-id' and '-fullpath'.
%   * Only .m files are scanned by default. Add other extensions via IncludeExt
%     at your own risk; CHECKCODE does not analyze non-.m sources.
%   * Compatible with MATLAB R2019b+ (older versions may return slightly
%     different CHECKCODE struct fields; this function handles that defensively).
%
%   See also: CHECKCODE, STRUCT2TABLE, WRITETABLE

% ---- Parse & validate inputs ------------------------------------------------
p = inputParser;
p.addRequired('targetPath', @(s)ischar(s) || (isstring(s) && isscalar(s)));
p.addParameter('Output', '', @(s)ischar(s) || isstring(s));
p.addParameter('Recursive', true, @(x)islogical(x) && isscalar(x));
p.addParameter('IncludeExt', {'.m'}, @(c)iscellstrlike(c));
p.addParameter('ExcludeDirs', defaultExcludeDirs(), @(c)iscellstrlike(c));
p.addParameter('ExcludeFiles', {}, @(c)iscellstrlike(c));
p.addParameter('Root', '', @(s)ischar(s) || isstring(s));
p.addParameter('OnError', 'record', @(s)any(strcmpi(s,{'record','rethrow'})));
p.addParameter('Quiet', true, @(x)islogical(x) && isscalar(x));
p.parse(targetPath, varargin{:});
opts = p.Results;

% Normalize strings
opts.Output       = char(string(opts.Output));
opts.Root         = char(string(opts.Root));
opts.targetPath   = char(string(targetPath));
includeExt        = lower(unique(cellfun(@(s)char(string(s)), opts.IncludeExt, 'UniformOutput', false)));
excludeDirs       = unique(cellfun(@(s)char(string(s)), opts.ExcludeDirs, 'UniformOutput', false));
excludeFiles      = unique(cellfun(@(s)char(string(s)), opts.ExcludeFiles, 'UniformOutput', false));

if isempty(opts.Root)
    if isfolder(opts.targetPath)
        opts.Root = char(java.io.File(opts.targetPath).getAbsolutePath());
    else
        opts.Root = fileparts(char(java.io.File(opts.targetPath).getAbsolutePath()));
    end
end
opts.Root = stripTrailingFilesep(opts.Root);

% ---- Build file list --------------------------------------------------------
fileList = {};
if isfolder(opts.targetPath)
    fileList = listFiles(opts.targetPath, includeExt, excludeDirs, excludeFiles, opts.Recursive);
else
    assert(exist(opts.targetPath,'file')==2, 'exportCodeIssues:NotFound', ...
        'File not found: %s', opts.targetPath);
    [~,~,ext] = fileparts(opts.targetPath);
    if ~any(strcmpi(ext, includeExt))
        warning('exportCodeIssues:ExtSkip', ...
            'File does not match IncludeExt; forcing include: %s', opts.targetPath);
    end
    fileList = {char(java.io.File(opts.targetPath).getAbsolutePath())};
end

if isempty(fileList)
    T = issuesTable();
    if ~isempty(opts.Output)
        writeOutput(T, opts.Output, opts.Root);
    end
    if ~opts.Quiet
        fprintf('[exportCodeIssues] No files found to analyze.\n');
    end
    return;
end

% ---- Run checkcode ----------------------------------------------------------
allIssues = issuesTable();
for i = 1:numel(fileList)
    f = fileList{i};
    if ~opts.Quiet
        fprintf('[exportCodeIssues] Analyzing %d/%d: %s\n', i, numel(fileList), f);
    end
    try
        % Request identifiers and absolute paths when available
        msgs = checkcode(f, '-id', '-fullpath'); %#ok<CHECKCODE>
    catch ME
        if strcmpi(opts.OnError,'rethrow')
            rethrow(ME);
        else
            % Record as a pseudo-issue so you see failures in the report
            row = table(string(f), string(relPath(f, opts.Root)), NaN, NaN, ...
                        string('CheckcodeError'), string(ME.message), ...
                        'VariableNames', {'File','RelFile','Line','Column','Identifier','Message'});
            allIssues = [allIssues; row]; %#ok<AGROW>
            continue;
        end
    end

    if isempty(msgs)
        % No issues for this file; optionally include a zero-issue row (skip by default)
        continue;
    end

    % msgs is a struct array; field names can vary across MATLAB versions.
    % Expected fields: message, line, column, identifier (if -id), file
    % Backfill missing fields defensively.
    msgs = ensureField(msgs, 'file', f);
    msgs = ensureField(msgs, 'line', NaN);
    msgs = ensureField(msgs, 'column', NaN);

    % Identifier can be missing if -id unsupported (very old MATLAB)
    if ~isfield(msgs, 'identifier')
        msgs = ensureField(msgs, 'identifier', extractIdentifierFallback(msgs));
    end

    % Build table rows
    n = numel(msgs);
    File       = strings(n,1);
    RelFile    = strings(n,1);
    Line       = nan(n,1);
    Column     = nan(n,1);
    Identifier = strings(n,1);
    Message    = strings(n,1);

    for k = 1:n
        File(k)       = string(absPathIfPossible(msgs(k), f));
        RelFile(k)    = string(relPath(File(k), opts.Root));
        Line(k)       = getNumericField(msgs(k), 'line');
        Column(k)     = getNumericField(msgs(k), 'column');
        Identifier(k) = string(getCharField(msgs(k), 'identifier'));
        Message(k)    = string(getCharField(msgs(k), 'message'));
    end

    allIssues = [allIssues; table(File, RelFile, Line, Column, Identifier, Message)]; %#ok<AGROW>
end

% Sort for stable, readable output
allIssues = sortrows(allIssues, {'RelFile','Line','Column','Identifier'});

% ---- Write output if requested ---------------------------------------------
if ~isempty(opts.Output)
    writeOutput(allIssues, opts.Output, opts.Root);
    if ~opts.Quiet
        fprintf('[exportCodeIssues] Wrote report: %s\n', opts.Output);
    end
end

% Return
T = allIssues;

end

% ======================= Local helpers =======================================
function tf = iscellstrlike(c)
    tf = iscell(c) && all(cellfun(@(s)ischar(s) || isstring(s), c));
end

function ex = defaultExcludeDirs()
    ex = {'.git','.svn','.hg','.venv','venv','env','.idea','.vs','node_modules', ...
          'slprj','codegen','mex','build','dist','external','third_party'};
end

function out = issuesTable()
    out = table(strings(0,1), strings(0,1), zeros(0,1), zeros(0,1), strings(0,1), strings(0,1), ...
                'VariableNames', {'File','RelFile','Line','Column','Identifier','Message'});
end

function list = listFiles(root, includeExt, excludeDirs, excludeFiles, recursive)
    root = stripTrailingFilesep(root);
    list = {};
    if ~recursive
        dd = dir(root);
        list = collectFilesFromDir(dd, root, includeExt, excludeFiles);
        return;
    end
    % Recursive descent
    stack = {root};
    while ~isempty(stack)
        d = stack{1}; stack(1) = [];
        dd = dir(d);
        % Add files in this dir
        list = [list, collectFilesFromDir(dd, d, includeExt, excludeFiles)]; %#ok<AGROW>
        % Queue subdirs
        subdirs = dd([dd.isdir]);
        for i = 1:numel(subdirs)
            name = subdirs(i).name;
            if strcmp(name,'.') || strcmp(name,'..')
                continue;
            end
            if any(strcmpi(name, excludeDirs))
                continue;
            end
            stack{end+1} = fullfile(d, name); %#ok<AGROW>
        end
    end
end

function files = collectFilesFromDir(dd, folder, includeExt, excludeFiles)
    files = {};
    dd = dd(~[dd.isdir]);
    for i = 1:numel(dd)
        [~, base, ext] = fileparts(dd(i).name);
        if ~any(strcmpi(ext, includeExt))
            continue;
        end
        full = fullfile(folder, [base, ext]);
        if any(cellfun(@(pat)~isempty(dir(fullfile(folder, pat))) && ~isempty(matchName(dd(i).name, pat)), excludeFiles)) %#ok<ISMAT>
            continue;
        end
        files{end+1} = char(java.io.File(full).getAbsolutePath()); %#ok<AGROW>
    end
end

function m = matchName(name, pattern)
    % Simple wildcard matcher supporting * and ?
    rx = regexptranslate('wildcard', pattern);
    m = ~isempty(regexp(name, ['^', rx, '$'], 'once'));
end

function s = stripTrailingFilesep(s)
    s = char(string(s));
    if ~isempty(s) && (s(end) == filesep)
        s(end) = [];
    end
end

function rel = relPath(absPath, root)
    absPath = char(string(absPath));
    root    = stripTrailingFilesep(root);
    if ispc
        % Case-insensitive on Windows
        if strncmpi(absPath, [root filesep], length(root)+1)
            rel = absPath((length(root)+2):end);
        else
            rel = absPath;
        end
    else
        if strncmp(absPath, [root filesep], length(root)+1)
            rel = absPath((length(root)+2):end);
        else
            rel = absPath;
        end
    end
end

function msgs = ensureField(msgs, fname, defaultVal)
    if ~isfield(msgs, fname)
        for k = 1:numel(msgs)
            msgs(k).(fname) = defaultVal; %#ok<AGROW>
        end
    end
end

function id = extractIdentifierFallback(msgs)
    % Try to extract something resembling an identifier from the message text
    id = repmat({''}, size(msgs));
    for k = 1:numel(msgs)
        m = getCharField(msgs(k), 'message');
        % Many messages start like: 'The value assigned to variable foo might be unused.'
        % We'll fall back to the first 32 chars as a pseudo-id if nothing else.
        if ~isempty(m)
            id{k} = ['MSG:', char(matlab.lang.makeValidName(m(1:min(32,end))))];
        else
            id{k} = 'MSG:Unknown';
        end
    end
end

function v = getNumericField(s, fn)
    % Return a scalar double for numeric-like fields; fallback to NaN.
    if isfield(s, fn)
        v = s.(fn);
        if isempty(v)
            v = NaN;
        else
            try
                % Coerce to scalar double; if vector, take first element.
                v = double(v(1));
            catch
                % Some older MATLAB versions may give unusual types; be safe.
                vv = str2double(string(v));
                if isnan(vv)
                    v = NaN;
                else
                    v = vv;
                end
            end
        end
    else
        v = NaN;
    end
end

function v = getCharField(s, fn)
    if isfield(s, fn)
        v = s.(fn);
        if isempty(v)
            v = '';
        else
            v = char(string(v));
        end
    else
        v = '';
    end
end

function f = absPathIfPossible(msg, fallback)
    if isfield(msg,'file') && ~isempty(msg.file)
        f = char(java.io.File(char(string(msg.file))).getAbsolutePath());
    else
        f = char(java.io.File(fallback).getAbsolutePath());
    end
end

function writeOutput(T, outPath, root)
    outPath = char(string(outPath));
    [outDir,~,ext] = fileparts(outPath);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end
    ext = lower(ext);
    switch ext
        case '.csv'
            writetable(T, outPath);
        case '.xlsx'
            writetable(T, outPath, 'FileType','spreadsheet');
        case '.json'
            % Convert to struct array with sensible field names
            S = table2struct(T);
            txt = jsonencode(S, 'PrettyPrint', true);
            fid = fopen(outPath, 'w', 'n','UTF-8');
            assert(fid>0, 'exportCodeIssues:IO', 'Could not open %s for writing.', outPath);
            cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
            fwrite(fid, txt, 'char');
        case '.md'
            fid = fopen(outPath, 'w', 'n','UTF-8');
            assert(fid>0, 'exportCodeIssues:IO', 'Could not open %s for writing.', outPath);
            cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
            fprintf(fid, '# Code Issues Report\n\n');
            fprintf(fid, '**Root:** %s\n\n', root);
            fprintf(fid, '| RelFile | Line | Column | Identifier | Message |\n');
            fprintf(fid, '|:--|--:|--:|:--|:--|\n');
            for i = 1:height(T)
                fprintf(fid, '| %s | %s | %s | %s | %s |\n', ...
                    escapeMd(T.RelFile(i)), num2strOrEmpty(T.Line(i)), ...
                    num2strOrEmpty(T.Column(i)), escapeMd(T.Identifier(i)), ...
                    escapeMd(T.Message(i)) );
            end
        otherwise
            error('exportCodeIssues:BadExt', 'Unsupported output extension: %s', ext);
    end
end

function s = escapeMd(str)
    s = char(str);
    s = strrep(s, '|', '\|');
    s = strrep(s, '\', '\\');
    s = regexprep(s, '[\r\n]+', ' ');
end

function s = num2strOrEmpty(v)
    if isnan(v)
        s = '';
    else
        s = num2str(v);
    end
end
