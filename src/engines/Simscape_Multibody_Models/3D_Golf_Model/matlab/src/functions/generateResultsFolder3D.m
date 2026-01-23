function generateResultsFolder3D(projectRoot, caseID)
% GENERATERESULTSFOLDER3D Creates a results folder structure and copies output files.
%   GENERATERESULTSFOLDER3D(projectRoot, caseID) creates a 'Model Output/<caseID>' folder
%   structure within the specified projectRoot and copies various files
%   (scripts, models, parameters, charts, tables) into the corresponding
%   subfolders and archives all scripts into a text file.

if nargin < 2
    % Default naming based on current date and hour-minute
    baseID = ['Run_' datestr(now, 'yyyymmdd_HHMM')];
    caseID = baseID;
    rootOut = fullfile(projectRoot, 'Model Output');

    % Check for folder collision and prompt user
    while exist(fullfile(rootOut, caseID), 'dir')
        fprintf('Case ID "%s" already exists.\n', caseID);
        caseID = input('Enter a new case ID: ', 's');
        if isempty(caseID)
            caseID = baseID;
        end
    end
end

warning('off', 'MATLAB:MKDIR:DirectoryExists');

% Define the main results folder path
resultsFolder = fullfile(projectRoot, 'Model Output', caseID);

% Create output folders
scriptsFolder = fullfile(resultsFolder, 'Scripts');
modelsFolder = fullfile(resultsFolder, 'Model and Parameters');
chartsFolder = fullfile(resultsFolder, 'Charts');
tablesFolder = fullfile(resultsFolder, 'Tables');
archiveFolder = fullfile(resultsFolder, 'ScriptArchives');

mkdir(resultsFolder);
mkdir(scriptsFolder);
mkdir(modelsFolder);
mkdir(chartsFolder);
mkdir(tablesFolder);
mkdir(archiveFolder);

fprintf('Creating results folder structure at: %s\n', resultsFolder);

% Define source and destination base paths
sourceScriptsBase = fullfile(projectRoot, 'Scripts');
sourceTablesBase = fullfile(projectRoot, 'Tables');
sourceModelsBase = projectRoot;

% Copy Chart Folders
chartSources = {
    fullfile(sourceScriptsBase, '_BaseData Scripts', 'BaseData Charts'),
    fullfile(sourceScriptsBase, '_BaseData Scripts', 'BaseData Quiver Plots'),
    fullfile(sourceScriptsBase, '_ZTCF Scripts', 'ZTCF Charts'),
    fullfile(sourceScriptsBase, '_ZTCF Scripts', 'ZTCF Quiver Plots'),
    fullfile(sourceScriptsBase, '_Delta Scripts', 'Delta Charts'),
    fullfile(sourceScriptsBase, '_Delta Scripts', 'Delta Quiver Plots'),
    fullfile(sourceScriptsBase, '_Comparison Scripts', 'Comparison Charts'),
    fullfile(sourceScriptsBase, '_Comparison Scripts', 'Comparison Quiver Plots'),
    fullfile(sourceScriptsBase, '_ZVCF Scripts', 'ZVCF Charts'),
    fullfile(sourceScriptsBase, '_ZVCF Scripts', 'ZVCF Quiver Plots')
};

for i = 1:length(chartSources)
    if exist(chartSources{i}, 'dir')
        fprintf('Copying charts from: %s\n', chartSources{i});
        copyfile(chartSources{i}, chartsFolder);
    else
        warning('Source chart folder not found: %s', chartSources{i});
    end
end

% Copy Model and Parameters files
modelSources = {
    fullfile(sourceModelsBase, 'GolfSwing3D_KineticallyDriven.slx'),
    fullfile(sourceModelsBase, 'ModelInputs.mat'),
    fullfile(sourceModelsBase, 'GolfSwing3D_ZVCF.slx'),
    fullfile(sourceModelsBase, 'ModelInputsZVCF.mat')
};

for i = 1:length(modelSources)
    if exist(modelSources{i}, 'file')
        fprintf('Copying model/parameter file: %s\n', modelSources{i});
        copyfile(modelSources{i}, modelsFolder);
    else
        warning('Source model/parameter file not found: %s', modelSources{i});
    end
end

% Copy Scripts folder (recursively)
if exist(sourceScriptsBase, 'dir')
    fprintf('Copying Scripts folder: %s\n', sourceScriptsBase);
    copyfile(sourceScriptsBase, scriptsFolder);
else
    warning('Source Scripts folder not found: %s', sourceScriptsBase);
end

% Copy Table files
tableSources = {
    fullfile(sourceTablesBase, 'BASEQ.mat'),
    fullfile(sourceTablesBase, 'ZTCFQ.mat'),
    fullfile(sourceTablesBase, 'DELTAQ.mat'),
    fullfile(sourceTablesBase, 'ZVCFTable.mat'),
    fullfile(sourceTablesBase, 'ZVCFTableQ.mat'),
    fullfile(sourceTablesBase, 'ClubQuiverAlphaReversal.mat'),
    fullfile(sourceTablesBase, 'ClubQuiverMaxCHS.mat'),
    fullfile(sourceTablesBase, 'ClubQuiverZTCFAlphaReversal.mat'),
    fullfile(sourceTablesBase, 'ClubQuiverDELTAAlphaReversal.mat'),
    fullfile(sourceTablesBase, 'SummaryTable.mat')
};

for i = 1:length(tableSources)
    if exist(tableSources{i}, 'file')
        fprintf('Copying table file: %s\n', tableSources{i});
        copyfile(tableSources{i}, tablesFolder);
    else
        warning('Source table file not found: %s', tableSources{i});
    end
end

% Archive all .m scripts into one .txt snapshot
timestamp = datestr(now, 'yyyy-mm-dd_HHMM');
archiveFile = fullfile(archiveFolder, [caseID '_Scripts_' timestamp '.txt']);
pack_project_no_prompt(projectRoot, archiveFile);

fprintf('File copying and script archiving to results folder complete.\n');
end
