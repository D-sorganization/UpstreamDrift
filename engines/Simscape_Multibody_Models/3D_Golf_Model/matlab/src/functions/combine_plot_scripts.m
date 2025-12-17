function combine_plot_scripts()
% COMBINE_PLOT_SCRIPTS - Consolidates multiple plot() scripts into one master file.
% Prompts user to select scripts from a folder, determines prefix from folder name,
% and replaces detected variables unless they already belong to known prefixes.

% Define known prefixes to preserve
knownPrefixes = {'BASEQ', 'DELTAQ', 'ZTCFQ', 'ZVCFQ', 'ModelData', 'DataTable'};

[fileNames, pathName] = uigetfile('*.m', 'Select PLOT scripts to combine', 'MultiSelect', 'on');
if isequal(fileNames, 0)
    disp('No files selected.');
    return;
end
if ischar(fileNames)
    fileNames = {fileNames};
end

[~, folderName] = fileparts(pathName);
caseName = upper(regexprep(folderName, '[^a-zA-Z0-9]', ''));
varPrefix = caseName;  % Assume variable name matches case name
outputFileName = fullfile(pathName, ['PLOT_' caseName '_Plots.m']);

header = sprintf([...
    'function PLOT_%s_Plots(%s)\n',...
    '%%% Auto-generated combined script\n',...
    '%%% Generated from %d files on %s\n\n'], ...
    caseName, varPrefix, numel(fileNames), datestr(now));

fidOut = fopen(outputFileName, 'w');
fprintf(fidOut, '%s', header);

for i = 1:length(fileNames)
    filePath = fullfile(pathName, fileNames{i});
    fprintf('\nProcessing %s...\n', fileNames{i});

    fprintf(fidOut, '%%%% ====== Start of %s ======\n', fileNames{i});

    lines = readlines(filePath);
    lines = strtrim(lines);
    lines = lines(~cellfun('isempty', lines));

    for j = 1:numel(lines)
        line = lines(j);

        if contains(line, 'clear') || contains(line, 'close all') || contains(line, 'clearvars')
            continue;
        end

        alreadyPrefixed = false;
        for k = 1:numel(knownPrefixes)
            if contains(line, [knownPrefixes{k}, '.'])
                alreadyPrefixed = true;
                break;
            end
        end

        if ~alreadyPrefixed
            line = regexprep(line, '\b([A-Z][A-Za-z0-9_]+)\(', [varPrefix, '.$1(']);
            line = regexprep(line, '\b([A-Z][A-Za-z0-9_]+)\.', [varPrefix, '.$1.']);
        end

        fprintf(fidOut, '%s\n', line);
    end

    fprintf(fidOut, '\n%%%% ====== End of %s ======\n\n', fileNames{i});
end

fprintf(fidOut, 'end\n');
fclose(fidOut);
disp(['Combined script saved to: ', outputFileName]);
end
