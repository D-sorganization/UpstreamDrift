function summary = run_frame_by_frame_torque_search(manifestFile)
%RUN_FRAME_BY_FRAME_TORQUE_SEARCH Sequential fallback torque search skeleton.
%
% summary = run_frame_by_frame_torque_search(manifestFile) reads a JSON
% manifest produced by prepare_frame_by_frame_search.py, evaluates
% short-horizon constant-torque candidates frame by frame, smooths the
% committed torque profile, and exports polynomial coefficients for the
% existing polynomial-input swing workflow.
%
% This file defines the workflow contract and deterministic orchestration.
% The actual stateful Simscape step is delegated to the +frame_search
% package: evaluateFrameByFrameTorqueCandidate, extractFrameByFrameState,
% and extractPredictedTarget all forward to frame_search.* helpers, which
% restore xFinal between frames and apply candidate torques as flat
% polynomial overrides on GolfSwing3D_Kinetic. See FRAME_BY_FRAME_OPTIMIZATION.md.

% Ensure the +frame_search package is reachable when this runner is called
% via an absolute path without the parent on the MATLAB path.
runnerFile = mfilename('fullpath');
matlabDir = fileparts(runnerFile);
if exist(fullfile(matlabDir, '+frame_search'), 'dir') && ~contains(path, matlabDir)
    addpath(matlabDir);
end

if nargin < 1 || isempty(manifestFile)
    mlDir = fileparts(matlabDir);
    manifestFile = fullfile( ...
        mlDir, 'data', 'processed', 'frame_by_frame_search.json');
end

if ~isfile(manifestFile)
    error('Frame-by-frame search manifest not found: %s', manifestFile);
end

config = jsondecode(fileread(manifestFile));
validateManifest(config);

targetTable = readtable(config.inputs.desired_target_csv);
time = targetTable.(config.columns.time);
controlColumns = jsonStringList(config.columns.control_columns);
targetColumns = jsonStringList(config.columns.target_columns);

candidateOffsets = buildCandidateOffsets( ...
    numel(controlColumns), config.search.candidate_levels, ...
    config.search.candidate_strategy);

useParallel = shouldUseParallel(config.search.use_parallel);
modelName = char(config.simulation.model_name);
currentState = initializeSearchState(config);
previousTorque = zeros(1, numel(controlColumns));
committedTorques = zeros(height(targetTable), numel(controlColumns));
frameScores = zeros(height(targetTable), 1);

if ~bdIsLoaded(modelName)
    load_system(modelName);
end

for frameIdx = 1:height(targetTable)
    targetFrame = table2struct(targetTable(frameIdx, targetColumns));
    candidateTorques = previousTorque + ...
        candidateOffsets .* config.search.candidate_step;
    scores = inf(size(candidateTorques, 1), 1);
    candidateStates = cell(size(candidateTorques, 1), 1);

    if useParallel
        parfor candidateIdx = 1:size(candidateTorques, 1)
            [scores(candidateIdx), candidateStates{candidateIdx}] = ...
                evaluateCandidate( ...
                    modelName, currentState, candidateTorques(candidateIdx, :), ...
                    targetFrame, config, controlColumns, previousTorque);
        end
    else
        for candidateIdx = 1:size(candidateTorques, 1)
            [scores(candidateIdx), candidateStates{candidateIdx}] = ...
                evaluateCandidate( ...
                    modelName, currentState, candidateTorques(candidateIdx, :), ...
                    targetFrame, config, controlColumns, previousTorque);
        end
    end

    [frameScores(frameIdx), bestIdx] = min(scores);
    previousTorque = candidateTorques(bestIdx, :);
    committedTorques(frameIdx, :) = previousTorque;
    currentState = candidateStates{bestIdx};
end

rawTorqueTable = array2table(committedTorques, 'VariableNames', controlColumns);
rawTorqueTable.time = time;
rawTorqueTable = movevars(rawTorqueTable, 'time', 'Before', 1);

smoothedTorqueTable = smoothTorqueTable( ...
    rawTorqueTable, controlColumns, config.postprocess.smoothing_window_frames);

writeTorqueOutputs(rawTorqueTable, smoothedTorqueTable, config);

summary = struct();
summary.manifestFile = manifestFile;
summary.frames = height(targetTable);
summary.candidatesPerFrame = size(candidateOffsets, 1);
summary.usedParallel = useParallel;
summary.bestFrameScores = frameScores;
summary.rawTorqueCsv = config.outputs.torque_csv;
summary.smoothedTorqueCsv = smoothedTorqueCsvPath(config.outputs.torque_csv);
summary.polynomialMat = config.outputs.polynomial_mat;
end

function values = jsonStringList(rawValues)
if iscell(rawValues)
    values = cellfun(@char, rawValues, 'UniformOutput', false);
elseif isstring(rawValues)
    values = cellstr(rawValues);
elseif ischar(rawValues)
    values = {rawValues};
else
    error('Expected JSON string list, got %s', class(rawValues));
end
end

function validateManifest(config)
requiredTop = {'inputs', 'simulation', 'columns', 'search', 'postprocess', 'outputs'};
for idx = 1:numel(requiredTop)
    if ~isfield(config, requiredTop{idx})
        error('Frame-by-frame manifest missing required section: %s', requiredTop{idx});
    end
end
if ~isfile(config.inputs.desired_target_csv)
    error('Desired target CSV not found: %s', config.inputs.desired_target_csv);
end
if config.search.horizon_frames < 1
    error('search.horizon_frames must be >= 1');
end
if config.search.candidate_step <= 0
    error('search.candidate_step must be positive');
end
end

function state = initializeSearchState(config)
state = struct();
state.starting_state_file = "";
if isfield(config.inputs, 'starting_state_file') && ...
        strlength(string(config.inputs.starting_state_file)) > 0
    if ~isfile(config.inputs.starting_state_file)
        error('Starting state file not found: %s', config.inputs.starting_state_file);
    end
    state.starting_state_file = string(config.inputs.starting_state_file);
end
state.frame_index = 0;
end

function offsets = buildCandidateOffsets(controlCount, levels, strategy)
levels = double(levels(:));
if strcmp(strategy, 'cartesian')
    grids = cell(1, controlCount);
    [grids{:}] = ndgrid(levels);
    offsets = zeros(numel(grids{1}), controlCount);
    for idx = 1:controlCount
        offsets(:, idx) = grids{idx}(:);
    end
    return;
end

if ~strcmp(strategy, 'coordinate')
    error('Unsupported candidate strategy: %s', strategy);
end

offsets = zeros(0, controlCount);
if any(levels == 0)
    offsets(end + 1, :) = zeros(1, controlCount);
end
nonZeroLevels = levels(levels ~= 0);
for controlIdx = 1:controlCount
    for levelIdx = 1:numel(nonZeroLevels)
        row = zeros(1, controlCount);
        row(controlIdx) = nonZeroLevels(levelIdx);
        offsets(end + 1, :) = row; %#ok<AGROW>
    end
end
end

function useParallel = shouldUseParallel(mode)
mode = char(mode);
if strcmp(mode, 'never')
    useParallel = false;
    return;
end

hasPCT = license('test', 'Distrib_Computing_Toolbox') && ...
    exist('gcp', 'file') == 2;
if strcmp(mode, 'always') && ~hasPCT
    error('Parallel requested, but Parallel Computing Toolbox is unavailable');
end
if ~hasPCT
    useParallel = false;
    return;
end

try
    gcp('nocreate');
    useParallel = true;
catch
    if strcmp(mode, 'always')
        rethrow(lasterror); %#ok<LERR>
    end
    useParallel = false;
end
end

function [score, nextState] = evaluateCandidate( ...
    modelName, currentState, torqueRow, targetFrame, config, ...
    controlColumns, previousTorque)
[simOut, candidateState] = evaluateFrameByFrameTorqueCandidate( ...
    modelName, currentState, torqueRow, targetFrame, config, controlColumns);
nextState = extractFrameByFrameState(simOut, candidateState, config);
score = scoreCandidate(simOut, torqueRow, targetFrame, config, previousTorque);
end

function [simOut, candidateState] = evaluateFrameByFrameTorqueCandidate( ...
    modelName, currentState, torqueRow, targetFrame, config, controlColumns)
%EVALUATEFRAMEBYFRAMETORQUECANDIDATE  GolfSwing3D_Kinetic candidate hook.
%
% Delegates to frame_search.evaluate_candidate_step which restores the
% previous frame's final state, applies torqueRow as a constant polynomial
% torque over the manifest's short horizon, and runs the model. The
% per-frame state hand-off is via Simulink SaveFinalState/InitialState
% (xFinal). Errors raised by the helper are propagated unchanged.
[simOut, candidateState] = frame_search.evaluate_candidate_step( ...
    modelName, currentState, torqueRow, targetFrame, config, controlColumns);
end

function nextState = extractFrameByFrameState(simOut, previousState, config)
%EXTRACTFRAMEBYFRAMESTATE  Pull final (q, qd, xFinal, time) into next state.
nextState = frame_search.extract_state(simOut, previousState, config);
end

function score = scoreCandidate(simOut, torqueRow, targetFrame, config, previousTorque)
predicted = extractPredictedTarget(simOut, targetFrame);
targetNames = fieldnames(targetFrame);
tracking = 0;
for idx = 1:numel(targetNames)
    name = targetNames{idx};
    if isfield(predicted, name)
        weight = targetWeight(name, config.search.weights);
        tracking = tracking + weight * (predicted.(name) - targetFrame.(name)) ^ 2;
    end
end
effort = config.search.weights.effort * sum(torqueRow .^ 2);
smoothness = config.search.weights.smoothness * sum((torqueRow - previousTorque) .^ 2);
score = tracking + effort + smoothness;
end

function predicted = extractPredictedTarget(simOut, targetFrame)
% Pull each manifest target column from the candidate simOut at the final
% sample. Missing columns raise frame_search:extract_predicted:missingColumn.
predicted = frame_search.extract_predicted(simOut, targetFrame);
end

function weight = targetWeight(columnName, weights)
if contains(columnName, 'Acceleration') || contains(columnName, '_a')
    weight = weights.acceleration;
elseif contains(columnName, 'Velocity') || contains(columnName, '_v')
    weight = weights.velocity;
else
    weight = weights.position;
end
end

function smoothed = smoothTorqueTable(rawTable, controlColumns, windowFrames)
smoothed = rawTable;
windowFrames = max(1, round(windowFrames));
for idx = 1:numel(controlColumns)
    name = controlColumns{idx};
    smoothed.(name) = smoothdata(rawTable.(name), 'movmean', windowFrames);
end
end

function writeTorqueOutputs(rawTorqueTable, smoothedTorqueTable, config)
rawPath = config.outputs.torque_csv;
smoothedPath = smoothedTorqueCsvPath(rawPath);
ensureParentDir(rawPath);
ensureParentDir(smoothedPath);
writetable(rawTorqueTable, rawPath);
writetable(smoothedTorqueTable, smoothedPath);
exportPolynomialCoefficients( ...
    smoothedTorqueTable, config.outputs.polynomial_mat, ...
    config.postprocess.polynomial_degree);
end

function path = smoothedTorqueCsvPath(rawPath)
[folder, name, ext] = fileparts(rawPath);
path = fullfile(folder, strcat(name, '_smoothed', ext));
end

function ensureParentDir(path)
folder = fileparts(path);
if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end
end

function exportPolynomialCoefficients(torqueTable, outputMat, degree)
time = torqueTable.time;
names = torqueTable.Properties.VariableNames;
payload = struct();
summary = struct();
summary.output_mat = outputMat;
summary.polynomial_degree = degree;
summary.fits = struct();

for idx = 1:numel(names)
    name = names{idx};
    if strcmp(name, 'time')
        continue;
    end
    coeffs = polyfit(time, torqueTable.(name), min(degree, height(torqueTable) - 1));
    matlabBase = torqueColumnToPolynomialBase(name);
    if strlength(matlabBase) == 0
        continue;
    end
    coeffs = padPolynomialCoefficients(coeffs, 6);
    letters = 'ABCDEFG';
    for coeffIdx = 1:numel(letters)
        payload.(strcat(matlabBase, letters(coeffIdx))) = coeffs(coeffIdx);
    end
    fit = struct();
    fit.matlab_base = char(matlabBase);
    fit.coefficients_A_to_G = coeffs;
    fit.rmse = sqrt(mean((polyval(coeffs, time) - torqueTable.(name)) .^ 2));
    summary.fits.(matlab.lang.makeValidName(name)) = fit;
end

ensureParentDir(outputMat);
save(outputMat, '-struct', 'payload');
summaryPath = replace(outputMat, '.mat', '.summary.json');
fid = fopen(summaryPath, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(summary, PrettyPrint=true));
end

function coeffs = padPolynomialCoefficients(coeffs, targetLength)
if numel(coeffs) < targetLength + 1
    coeffs = [zeros(1, targetLength + 1 - numel(coeffs)), coeffs];
end
end

function matlabBase = torqueColumnToPolynomialBase(name)
mapping = struct( ...
    'LScapLogs_ActuatorTorqueX', 'LScapInputX', ...
    'LScapLogs_ActuatorTorqueY', 'LScapInputY', ...
    'RScapLogs_ActuatorTorqueX', 'RScapInputX', ...
    'RScapLogs_ActuatorTorqueY', 'RScapInputY', ...
    'LSLogs_ActuatorTorqueX', 'LSInputX', ...
    'LSLogs_ActuatorTorqueY', 'LSInputY', ...
    'LSLogs_ActuatorTorqueZ', 'LSInputZ', ...
    'RSLogs_ActuatorTorqueX', 'RSInputX', ...
    'RSLogs_ActuatorTorqueY', 'RSInputY', ...
    'RSLogs_ActuatorTorqueZ', 'RSInputZ', ...
    'SpineLogs_ActuatorTorqueX', 'SpineInputX', ...
    'SpineLogs_ActuatorTorqueY', 'SpineInputY', ...
    'HipLogs_TranslationForceXInput', 'TranslationInputX', ...
    'HipLogs_TranslationForceYInput', 'TranslationInputY', ...
    'HipLogs_TranslationForceZInput', 'TranslationInputZ', ...
    'HipLogs_HipTorqueXInput', 'HipInputX', ...
    'HipLogs_HipTorqueYInput', 'HipInputY', ...
    'HipLogs_HipTorqueZInput', 'HipInputZ');
validName = matlab.lang.makeValidName(name);
if isfield(mapping, validName)
    matlabBase = string(mapping.(validName));
else
    matlabBase = "";
end
end
