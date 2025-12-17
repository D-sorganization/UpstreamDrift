%Turn off the warning that a directory already exists when you create it.
warning('off', 'MATLAB:MKDIR:DirectoryExists');

%Create a folder named results with organized plots and backup data to
%rerun each trial is located:
cd(matlabdrive);
cd '2DModel';
mkdir 'Model Output';
cd(matlabdrive);
cd '2DModel';
cd 'Model Output';
mkdir 'Scripts';
cd(matlabdrive);
cd '2DModel';
cd 'Model Output';
mkdir 'Model and Parameters';
cd(matlabdrive);
cd '2DModel';
cd 'Model Output';
mkdir 'Charts';
cd(matlabdrive);
cd '2DModel';
cd 'Model Output';
mkdir 'Tables';

%Write to the charts folder: How do I make these copy file from a location
%specified in matlab drive using the matlab function?
cd(matlabdrive);
cd '2DModel';
copyfile 'Scripts/_BaseData Scripts/BaseData Charts' 'Model Output/Charts/';
copyfile 'Scripts/_BaseData Scripts/BaseData Quiver Plots' 'Model Output/Charts/';
copyfile 'Scripts/_ZTCF Scripts/ZTCF Charts' 'Model Output/Charts/';
copyfile 'Scripts/_ZTCF Scripts/ZTCF Quiver Plots' 'Model Output/Charts/';
copyfile 'Scripts/_Delta Scripts/Delta Charts' 'Model Output/Charts/';
copyfile 'Scripts/_Delta Scripts/Delta Quiver Plots' 'Model Output/Charts/';
copyfile 'Scripts/_Comparison Scripts/Comparison Charts' 'Model Output/Charts/';
copyfile 'Scripts/_Comparison Scripts/Comparison Quiver Plots' 'Model Output/Charts/';
copyfile 'Scripts/_ZVCF Scripts/ZVCF Charts' 'Model Output/Charts/';
copyfile 'Scripts/_ZVCF Scripts/ZVCF Quiver Plots' 'Model Output/Charts/';


%Write to the Model and Parameters folder:
cd(matlabdrive);
cd '2DModel';
copyfile 'GolfSwing.slx' 'Model Output/Model and Parameters';
copyfile 'ModelInputs.mat' 'Model Output/Model and Parameters';
copyfile 'GolfSwingZVCF.slx' 'Model Output/Model and Parameters';
copyfile 'ModelInputsZVCF.mat' 'Model Output/Model and Parameters';

%Write to the Scripts folder:
cd(matlabdrive);
cd '2DModel';
copyfile 'Scripts/' 'Model Output/Scripts';

%Write to the Tables folder on Model Output
cd(matlabdrive);
cd '2DModel';
copyfile 'Tables/BASE.mat' 'Model Output/Tables/';
copyfile 'Tables/ZTCF.mat' 'Model Output/Tables/';
copyfile 'Tables/DELTA.mat' 'Model Output/Tables/';
copyfile 'Tables/BASEQ.mat' 'Model Output/Tables/';
copyfile 'Tables/ZTCFQ.mat' 'Model Output/Tables/';
copyfile 'Tables/DELTAQ.mat' 'Model Output/Tables/';
copyfile 'Tables/ZVCFTable.mat' 'Model Output/Tables/';
copyfile 'Tables/ZVCFTableQ.mat' 'Model Output/Tables/';
copyfile 'Tables/ClubQuiverAlphaReversal.mat' 'Model Output/Tables/';
copyfile 'Tables/ClubQuiverMaxCHS.mat' 'Model Output/Tables/';
copyfile 'Tables/SummaryTable.mat' 'Model Output/Tables/';

cd(matlabdrive);
cd '2DModel';