function export_simulated_club_csv(simOut, outputCsvFile)
%EXPORT_SIMULATED_CLUB_CSV Export club-head logs from a simulation result.
%
% The comparison and calibration scripts consume a compact CSV with time and
% ClubLogs_CHGlobalPosition/Velocity/Acceleration columns.

if nargin < 2 || isempty(outputCsvFile)
    thisFile = mfilename('fullpath');
    mlDir = fileparts(fileparts(thisFile));
    outputCsvFile = fullfile(mlDir, 'data', 'processed', 'simulated_club_motion.csv');
end

logs = simOut.logsout;
if isempty(logs)
    error('Simulation output does not contain logsout.');
end

position = extractElement(logs, 'ClubLogs_CHGlobalPosition');
velocity = extractElement(logs, 'ClubLogs_CHGlobalVelocity');
acceleration = extractElement(logs, 'ClubLogs_CHGlobalAcceleration');

time = position.Time(:);
tableData = table(time, 'VariableNames', {'time'});
tableData = appendVector(tableData, position.Data, 'ClubLogs_CHGlobalPosition');
tableData = appendVector(tableData, velocity.Data, 'ClubLogs_CHGlobalVelocity');
if ~isempty(acceleration)
    tableData = appendVector(tableData, acceleration.Data, 'ClubLogs_CHGlobalAcceleration');
end

outputDir = fileparts(outputCsvFile);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
writetable(tableData, outputCsvFile);
end

function timeseriesData = extractElement(logs, elementName)
element = logs.find(elementName);
if isempty(element)
    timeseriesData = [];
    return;
end
timeseriesData = element.Values;
end

function tableData = appendVector(tableData, values, baseName)
values = squeeze(values);
if isvector(values)
    values = values(:);
end
if size(values, 1) ~= height(tableData)
    values = values';
end
for idx = 1:size(values, 2)
    tableData.(sprintf('%s_%d', baseName, idx)) = values(:, idx);
end
end
