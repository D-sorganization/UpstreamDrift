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
% The actual stateful Simscape step is intentionally an extension point:
% implement evaluateFrameByFrameTorqueCandidate and extractFrameByFrameState
% for GolfSwing3D_Kinetic before using this for production overnight runs.

if nargin < 1 || isempty(manifestFile)
    thisFile = mfilename('fullpath');
    mlDir = fileparts(fileparts(thisFile));
    manifestFile = fullfile( ...
        mlDir, 'data', 'processed', 'frame_by_frame_search.json');
end

if ~isfile(manifestFile)
    error('Frame-by-frame search manifest not found: %s', manifestFile);
end

manifestBytes = fileread(manifestFile);
config = jsondecode(manifestBytes);
validateManifest(config);
manifestSha = computeManifestSha256(manifestFile);

targetTable = readtable(config.inputs.desired_target_csv);
time = targetTable.(config.columns.time);
controlColumns = jsonStringList(config.columns.control_columns);
targetColumns = jsonStringList(config.columns.target_columns);

candidateOffsets = buildCandidateOffsets( ...
    numel(controlColumns), config.search.candidate_levels, ...
    config.search.candidate_strategy);

useParallel = shouldUseParallel(config.search.use_parallel);
modelName = char(config.simulation.model_name);
runDir = char(config.outputs.run_dir);
progressCsv = char(config.outputs.progress_csv);
checkpointInterval = checkpointIntervalFrames(config);
expectedFrameSeconds = expectedFrameWallClock(config);
staleMultiplier = staleLockMultiplier(config);

if ~exist(runDir, 'dir')
    mkdir(runDir);
end

[resumeState, resumed] = frame_search.resume( ...
    runDir, manifestSha, expectedFrameSeconds, staleMultiplier);

currentState = initializeSearchState(config);
previousTorque = zeros(1, numel(controlColumns));
committedTorques = zeros(height(targetTable), numel(controlColumns));
frameScores = zeros(height(targetTable), 1);
wallClockPerFrame = zeros(height(targetTable), 1);
startFrame = 1;

if resumed
    startFrame = resumeState.last_frame_idx + 1;
    nKeep = min(resumeState.last_frame_idx, height(targetTable));
    committedTorques(1:nKeep, :) = resumeState.committed_torques(1:nKeep, :);
    frameScores(1:nKeep) = resumeState.frame_scores(1:nKeep);
    if isfield(resumeState, 'wall_clock_per_frame')
        nWc = min(numel(resumeState.wall_clock_per_frame), height(targetTable));
        wallClockPerFrame(1:nWc) = resumeState.wall_clock_per_frame(1:nWc);
    end
    previousTorque = resumeState.previous_torque;
    currentState = resumeState.current_state;
    fprintf('Resumed frame-search at frame %d/%d\n', ...
        startFrame, height(targetTable));
else
    initializeProgressCsv(progressCsv);
end

if ~bdIsLoaded(modelName)
    load_system(modelName);
end

for frameIdx = startFrame:height(targetTable)
    frameTic = tic;
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
    wallClockPerFrame(frameIdx) = toc(frameTic);

    appendProgressRow(progressCsv, frameIdx, bestIdx, ...
        frameScores(frameIdx), wallClockPerFrame(frameIdx));

    if mod(frameIdx, checkpointInterval) == 0 || frameIdx == height(targetTable)
        chkState = struct( ...
            'manifest_sha256', manifestSha, ...
            'last_frame_idx', frameIdx, ...
            'previous_torque', previousTorque, ...
            'current_state', currentState, ...
            'committed_torques', committedTorques, ...
            'frame_scores', frameScores, ...
            'wall_clock_per_frame', wallClockPerFrame);
        frame_search.checkpoint(runDir, chkState);
    end
end

rawTorqueTable = array2table(committedTorques, 'VariableNames', controlColumns);
rawTorqueTable.time = time;
rawTorqueTable = movevars(rawTorqueTable, 'time', 'Before', 1);

smoothedTorqueTable = smoothTorqueTable( ...
    rawTorqueTable, controlColumns, config.postprocess.smoothing_window_frames);

writeTorqueOutputs(rawTorqueTable, smoothedTorqueTable, config);

summary = struct();
summary.manifestFile = manifestFile;
summary.manifestSha256 = manifestSha;
summary.frames = height(targetTable);
summary.candidatesPerFrame = size(candidateOffsets, 1);
summary.usedParallel = useParallel;
summary.bestFrameScores = frameScores;
summary.rawTorqueCsv = config.outputs.torque_csv;
summary.smoothedTorqueCsv = smoothedTorqueCsvPath(config.outputs.torque_csv);
summary.polynomialMat = config.outputs.polynomial_mat;
summary.runDir = runDir;
summary.progressCsv = progressCsv;
summary.checkpointMat = char(config.outputs.checkpoint_mat);
summary.totalWallClockSeconds = sum(wallClockPerFrame);
summary.resumed = resumed;

writeRunSummary(config, summary);
end

function sha = computeManifestSha256(manifestFile)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(manifestFile, 'r');
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
digest = md.digest(bytes);
hexParts = cell(1, numel(digest));
for idx = 1:numel(digest)
    hexParts{idx} = sprintf('%02x', typecast(digest(idx), 'uint8'));
end
sha = strjoin(hexParts, '');
end

function interval = checkpointIntervalFrames(config)
interval = 10;
if isfield(config, 'checkpoint') && isfield(config.checkpoint, 'interval_frames')
    interval = double(config.checkpoint.interval_frames);
end
if interval < 1
    interval = 1;
end
end

function multiplier = staleLockMultiplier(config)
multiplier = 2.0;
if isfield(config, 'checkpoint') && isfield(config.checkpoint, 'stale_lock_multiplier')
    multiplier = double(config.checkpoint.stale_lock_multiplier);
end
end

function seconds = expectedFrameWallClock(config)
seconds = 0;
if isfield(config, 'validation') && isfield(config.validation, 'median_step_seconds')
    seconds = double(config.validation.median_step_seconds);
end
end

function initializeProgressCsv(progressCsv)
folder = fileparts(progressCsv);
if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end
fid = fopen(progressCsv, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'frame_idx,selected_candidate,score,wall_clock_s,timestamp\n');
end

function appendProgressRow(progressCsv, frameIdx, candidateIdx, score, wallClock)
fid = fopen(progressCsv, 'a');
cleanup = onCleanup(@() fclose(fid));
ts = datestr(now, 'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>
fprintf(fid, '%d,%d,%.10g,%.6f,%s\n', ...
    frameIdx, candidateIdx, score, wallClock, ts);
end

function writeRunSummary(config, summary)
summaryPath = char(config.outputs.summary_json);
folder = fileparts(summaryPath);
if ~isempty(folder) && ~exist(folder, 'dir')
    mkdir(folder);
end
payload = struct( ...
    'manifest_sha256', summary.manifestSha256, ...
    'total_frames', summary.frames, ...
    'candidate_evaluations', summary.frames * summary.candidatesPerFrame, ...
    'total_wall_clock_seconds', summary.totalWallClockSeconds, ...
    'used_parallel', summary.usedParallel, ...
    'resumed', summary.resumed, ...
    'progress_csv', summary.progressCsv, ...
    'checkpoint_mat', summary.checkpointMat, ...
    'raw_torque_csv', summary.rawTorqueCsv, ...
    'smoothed_torque_csv', summary.smoothedTorqueCsv, ...
    'polynomial_mat', summary.polynomialMat);
fid = fopen(summaryPath, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(payload, PrettyPrint=true));
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
    modelName, currentState, torqueRow, targetFrame, config, controlColumns) %#ok<INUSD>
simOut = []; %#ok<NASGU>
candidateState = currentState; %#ok<NASGU>
error([ ...
    'Frame-by-frame Simscape stepping is not implemented yet. ', ...
    'Implement evaluateFrameByFrameTorqueCandidate for GolfSwing3D_Kinetic ', ...
    'to restore currentState, apply torqueRow over the configured horizon, ', ...
    'and return the candidate simOut.']);
end

function nextState = extractFrameByFrameState(simOut, previousState, config) %#ok<INUSD>
nextState = previousState;
if isstruct(nextState) && isfield(nextState, 'frame_index')
    nextState.frame_index = nextState.frame_index + 1;
end
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

function predicted = extractPredictedTarget(simOut, targetFrame) %#ok<INUSD>
% Extension point: map logged Simscape outputs to targetFrame field names.
% The default returns an empty struct so production runs cannot silently claim
% good tracking without a model-specific implementation.
predicted = struct();
if isempty(fieldnames(predicted))
    error([ ...
        'Target extraction is not implemented yet. Map Simscape logs to the ', ...
        'desired club/body target columns before running the search.']);
end
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
