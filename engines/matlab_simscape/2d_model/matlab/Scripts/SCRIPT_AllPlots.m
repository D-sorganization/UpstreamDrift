%Master Script Plot:
PauseTime=0;

cd(matlabdrive);
cd '2DModel/Scripts/_BaseData Scripts'/;
MASTER_SCRIPT_BaseDataCharts;

cd(matlabdrive);
cd '2DModel/Scripts/_ZTCF Scripts';
MASTER_SCRIPT_ZTCFCharts;

cd(matlabdrive);
cd '2DModel/Scripts/_Delta Scripts';
MASTER_SCRIPT_DeltaCharts;

cd(matlabdrive);
cd '2DModel/Scripts/_Comparison Scripts';
MASTER_SCRIPT_ComparisonCharts;

cd(matlabdrive);
cd '2DModel/Scripts/';
SCRIPT_ResultsFolderGeneration;

cd(matlabdrive);
cd '2DModel/Scripts/_ZVCF Scripts';
MASTER_SCRIPT_ZVCF_CHARTS;

clear PauseTime;