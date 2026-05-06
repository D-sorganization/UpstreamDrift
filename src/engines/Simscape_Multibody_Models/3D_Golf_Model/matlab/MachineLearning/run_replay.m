function run_replay(polynomialMatFile, scenario, startStateMatFile, simCsvOut, jointVelocityCsvOut)
%RUN_REPLAY Closed-loop MATLAB replay driver for issue #3970.
%
% run_replay(polynomialMatFile, scenario, startStateMatFile, simCsvOut, jointVelocityCsvOut)
%
% Drives the full replay loop:
%   1. Optionally exports the start state from the scenario-specific input MAT
%      when ``startStateMatFile`` is empty.
%   2. Runs ``GolfSwing3D_Kinetic`` via ``run_ml_polynomial_input_swing`` with
%      the supplied polynomial coefficient MAT.
%   3. Exports the simulated club CSV via ``export_simulated_club_csv``.
%   4. Best-effort exports a joint-velocity CSV when ``logsout`` carries
%      angular velocity signals.
%
% Inputs
%   polynomialMatFile   absolute path to fitted polynomial coefficient MAT
%   scenario            'full-swing' or 'downswing'
%   startStateMatFile   optional absolute path to a start-state MAT; when
%                       empty, one is generated from the scenario default
%   simCsvOut           absolute output path for simulated club CSV
%   jointVelocityCsvOut absolute output path for joint-velocity CSV
%
% Errors propagate as MATLAB exceptions; the Python harness translates them
% into ``ReplayError`` for the caller.

if nargin < 5
    error('run_replay:Args', ...
        'usage: run_replay(polynomialMat, scenario, startStateMat, simCsv, jointVelCsv)');
end

if ~ischar(scenario) && ~isstring(scenario)
    error('run_replay:Args', 'scenario must be a string');
end
scenario = char(scenario);

if ~isfile(polynomialMatFile)
    error('run_replay:MissingInput', ...
        'Polynomial coefficient MAT not found: %s', polynomialMatFile);
end

thisDir = fileparts(mfilename('fullpath'));
mlMatlabDir = fullfile(fileparts(fileparts(thisDir)), ...
    'MachineLearning', 'matlab');
if exist(mlMatlabDir, 'dir')
    addpath(mlMatlabDir);
end

if isempty(startStateMatFile)
    startStateMatFile = fullfile(tempdir, ...
        sprintf('ml_replay_start_state_%s.mat', sanitizeScenario(scenario)));
    export_start_state_from_input_file(scenario, startStateMatFile);
elseif ~isfile(startStateMatFile)
    error('run_replay:MissingInput', ...
        'Start-state MAT not found: %s', startStateMatFile);
end

simOut = run_ml_polynomial_input_swing( ...
    polynomialMatFile, 'GolfSwing3D_Kinetic', startStateMatFile);

simOutDir = fileparts(simCsvOut);
if ~isempty(simOutDir) && ~exist(simOutDir, 'dir')
    mkdir(simOutDir);
end
export_simulated_club_csv(simOut, simCsvOut);

if ~isempty(jointVelocityCsvOut)
    try
        exportJointVelocity(simOut, jointVelocityCsvOut);
    catch jvErr
        warning('run_replay:JointVelocity', ...
            'Joint-velocity export failed: %s', jvErr.message);
    end
end
end

function exportJointVelocity(simOut, outputCsv)
logs = simOut.logsout;
if isempty(logs)
    return;
end

names = {};
data = {};
time = [];
for idx = 1:logs.numElements
    elem = logs.getElement(idx);
    name = elem.Name;
    if isempty(regexpi(name, 'velocity', 'once'))
        continue;
    end
    values = squeeze(elem.Values.Data);
    if isvector(values)
        values = values(:);
    elseif size(values, 1) ~= numel(elem.Values.Time)
        values = values';
    end
    if isempty(time)
        time = elem.Values.Time(:);
    end
    for axis = 1:size(values, 2)
        names{end + 1} = sprintf('%s_%d', name, axis); %#ok<AGROW>
        data{end + 1} = values(:, axis); %#ok<AGROW>
    end
end

if isempty(names)
    return;
end

outDir = fileparts(outputCsv);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

tableData = table(time, 'VariableNames', {'time'});
for idx = 1:numel(names)
    tableData.(names{idx}) = data{idx};
end
writetable(tableData, outputCsv);
end

function value = sanitizeScenario(scenario)
value = regexprep(char(string(scenario)), '[^A-Za-z0-9]+', '_');
value = regexprep(value, '^_+|_+$', '');
end
