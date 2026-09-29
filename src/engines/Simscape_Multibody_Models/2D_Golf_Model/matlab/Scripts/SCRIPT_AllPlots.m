%Master Script Plot:
PauseTime=0;

scriptDir = fileparts(mfilename('fullpath'));

cd(fullfile(scriptDir, '_BaseData Scripts'));
MASTER_SCRIPT_BaseDataCharts;

cd(fullfile(scriptDir, '_ZTCF Scripts'));
MASTER_SCRIPT_ZTCFCharts;

cd(fullfile(scriptDir, '_Delta Scripts'));
MASTER_SCRIPT_DeltaCharts;

cd(fullfile(scriptDir, '_Comparison Scripts'));
MASTER_SCRIPT_ComparisonCharts;

cd(scriptDir);
SCRIPT_ResultsFolderGeneration;

cd(fullfile(scriptDir, '_ZVCF Scripts'));
MASTER_SCRIPT_ZVCF_CHARTS;

clear PauseTime;
