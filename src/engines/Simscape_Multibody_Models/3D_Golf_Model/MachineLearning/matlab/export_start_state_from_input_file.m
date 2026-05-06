function exportedNames = export_start_state_from_input_file(scenario, outputMatFile, sourceMatFile)
%EXPORT_START_STATE_FROM_INPUT_FILE Export model start positions/velocities.
%
% exportedNames = export_start_state_from_input_file("full-swing", outputMatFile)
% exports address-style start variables from 3DModelInputs.mat.
%
% exportedNames = export_start_state_from_input_file("downswing", outputMatFile)
% exports top-of-backswing start variables from 3DModelInputs_TopofBackswing.mat.
%
% The output MAT is intended to be passed as the third argument to
% run_ml_polynomial_input_swing, keeping start-state selection independent of
% the ML polynomial torque coefficient file.

if nargin < 1 || isempty(scenario)
    scenario = "full-swing";
end

if nargin < 2 || isempty(outputMatFile)
    thisFile = mfilename('fullpath');
    mlDir = fileparts(fileparts(thisFile));
    outputMatFile = fullfile(mlDir, 'data', 'processed', ...
        sprintf('ml_%s_start_state.mat', sanitizeScenario(scenario)));
end

modelInputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    '..', 'matlab', 'src', 'model', 'inputs');

if nargin < 3 || isempty(sourceMatFile)
    switch string(scenario)
        case "downswing"
            sourceMatFile = fullfile(modelInputDir, '3DModelInputs_TopofBackswing.mat');
        otherwise
            sourceMatFile = fullfile(modelInputDir, '3DModelInputs.mat');
    end
end

if ~isfile(sourceMatFile)
    error('Source model input MAT file not found: %s', sourceMatFile);
end

sourceData = load(sourceMatFile);
names = fieldnames(sourceData);
exportData = struct();
exportedNames = {};

for idx = 1:numel(names)
    name = names{idx};
    if isempty(regexp(name, 'Start(Position|Velocity)', 'once'))
        continue;
    end
    value = sourceData.(name);
    if isa(value, 'Simulink.Parameter')
        value = value.Value;
    end
    if isnumeric(value) && isscalar(value)
        exportData.(name) = double(value);
        exportedNames{end + 1} = name; %#ok<AGROW>
    end
end

if isempty(exportedNames)
    error('No scalar start-state variables were found in %s', sourceMatFile);
end

outputDir = fileparts(outputMatFile);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(outputMatFile, '-struct', 'exportData');

summaryFile = replace(outputMatFile, '.mat', '.summary.txt');
fid = fopen(summaryFile, 'w');
if fid >= 0
    fprintf(fid, 'scenario: %s\n', string(scenario));
    fprintf(fid, 'source: %s\n', sourceMatFile);
    fprintf(fid, 'output: %s\n', outputMatFile);
    fprintf(fid, 'variables: %d\n', numel(exportedNames));
    for idx = 1:numel(exportedNames)
        fprintf(fid, '%s\n', exportedNames{idx});
    end
    fclose(fid);
end
end

function value = sanitizeScenario(scenario)
value = regexprep(char(string(scenario)), '[^A-Za-z0-9]+', '_');
value = regexprep(value, '^_+|_+$', '');
end
