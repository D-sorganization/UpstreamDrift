% ZTCF and ZVCF Data Generation Script
% The ZTCF is the Zero Torque Counterfactual. It represents the passive
% effect of all things that are not joint torques on the interaction forces
% within the system. It is calculated by step changing the forces to zero
% at time varying points in the model. The interaction forces are
% determined for the time in which the model has the same position and
% velocity as in the swing but the joint torques are zero.

cd(matlabdrive);
cd '2DModel';
GolfSwing

cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_mdlWks_Generate;

% Turn on / off dampening in the killswitch
assignin(mdlWks,'KillDampFinalValue',Simulink.Parameter(0)) % Dampening Included in Killswitch
%assignin(mdlWks,'KillDampFinalValue',Simulink.Parameter(1)) % Dampening Excluded in Killswitch

%Set the stop time to 0.28 seconds so table generation works in the loop
assignin(mdlWks,'StopTime',Simulink.Parameter(0.28));

%Turn off the warning that a directory already exists when you create it.
warning('off', 'MATLAB:MKDIR:DirectoryExists');
warning off Simulink:Masking:NonTunableParameterChangedDuringSimulation;
warning off Simulink:Engine:NonTunableVarChangedInFastRestart;
warning off Simulink:Engine:NonTunableVarChangedMaxWarnings;

% mdlWks Generation File Creation
mdlWks=get_param('GolfSwing','ModelWorkspace');

%Set the killswitch time to 1 second so it doesn't ever trigger
assignin(mdlWks,'KillswitchStepTime',Simulink.Parameter(1));

% Set up the model to return a single matlab object as output.
set_param(GolfSwing,"ReturnWorkspaceOutputs","on");
set_param(GolfSwing,"FastRestart","on");
set_param(GolfSwing,"MaxStep","0.001");

% Run the model to generate BaseData table. Save model output as "out".
out=sim(GolfSwing);

% Run Table Generation Script on "out"
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_TableGeneration;

% Create table called BaseData with the "Data" Table from run with no
% killswitch.
BaseData=Data;

% Copy Data to ZTCF Table to Get Variable Names
ZTCFTable=Data; %Create a table called ZTCFTable from Data.
ZTCFTable(:,:)=[]; %Delete All Data in ZTCF Table and Replace with Blanks

% Begin Generation of ZTCF Data by Looping

%for i=0:280
for i=0:28    
     
    %Scale counter to match desired times
    %j=i/1000;
    j=i/100;

    %Display Percentage
    %ZTCFPercentComplete=i/280*100
    ZTCFPercentComplete=i/28*100
   
    %Write step time to model workspace
    assignin(mdlWks,'KillswitchStepTime',Simulink.Parameter(j));     
    out=sim(GolfSwing);
    SCRIPT_TableGeneration;
    ZTCFData=Data;
    
    %Find the row where the KillswitchState first becomes zero  
    row=find(ZTCFData.KillswitchState==0,1);

    %Copy ZTCF Data Table to ZTCF
    ZTCF=ZTCFData;
    
    %Rewrite first row in ZTCF to the values in "row" calculated above
    ZTCF(1,:)=ZTCFData(row,:);
    
    %Find height of table and subtract 1 (number of rows to delete)
    H=height(ZTCF)-1;

    %Clear all rows from 2 to height of table using a loop
    for k=1:H;
        DelRow=H+2-k;
        ZTCF(DelRow,:)=[];
    end

    clear k;
    clear H;
    clear DelRow;
    
    %This has generated the table row and labels for ZTCF. Now it needs to
    %get compiled and added to the ZTCFTable generated earlier
    ZTCFTable=[ZTCFTable;ZTCF];

end
    
%Cleanup Workspace and Only Leave Important Stuff
clear j;
clear out;
clear row;
clear steptime;
clear Data;
clear ZTCF;
clear ZTCFData;
ZTCF=ZTCFTable;
clear ZTCFTable;

% Reset the killswitch time to 1
assignin(mdlWks,'KillswitchStepTime',Simulink.Parameter(1)); 

% Make timetable formats with BaseData and ZTCF Tables to put them in a
% format where I can use the retime() function

%Generate duration times for each table:
BaseDataTime=seconds(BaseData.Time);
ZTCFTime=seconds(ZTCF.Time);

%Create versions of the tables to modify:
BaseDataTemp=BaseData; 
ZTCFTemp=ZTCF;

%Write duration times into the tables:
BaseDataTemp.('t')=BaseDataTime;
ZTCFTemp.('t')=ZTCFTime;

%Create a timetable using the duration timnes generated above:
BaseDataTimetableTemp=table2timetable(BaseDataTemp,"RowTimes","t");
ZTCFTimetableTemp=table2timetable(ZTCFTemp,"RowTimes","t");

%Remove the remaining time variable from the Timetable:
BaseDataTimetable=removevars(BaseDataTimetableTemp,'Time');
ZTCFTimetable=removevars(ZTCFTimetableTemp,'Time');

%Generate a matched set of base data to ZTCF data using interpolation using the retime()
%function:
BaseDataMatched=retime(BaseDataTimetable,ZTCFTime,'spline');

%Generate the Delta:
DELTATimetable=BaseDataMatched-ZTCFTimetable;

%Define sample time of new table
Ts=0.0001;

BASEUniform=retime(BaseDataMatched,'regular','spline','TimeStep',seconds(Ts));
ZTCFUniform=retime(ZTCFTimetable,'regular','spline','TimeStep',seconds(Ts));
DELTAUniform=retime(DELTATimetable,'regular','spline','TimeStep',seconds(Ts));

%Cleanup a bit
clear BaseDataTimetable;
clear BaseDataTemp;
clear BaseDataTime;
clear BaseDataTimetableTemp;
clear ZTCFTimetable
clear ZTCFTemp;
clear ZTCFTime;
clear ZTCFTimetableTemp;

%Convert to Table:
DELTA=timetable2table(DELTAUniform,"ConvertRowTimes",true);
DELTA=renamevars(DELTA,"t","Time");

BASE=timetable2table(BASEUniform,"ConvertRowTimes",true);
BASE=renamevars(BASE,"t","Time");

ZTCF=timetable2table(ZTCFUniform,"ConvertRowTimes",true);
ZTCF=renamevars(ZTCF,"t","Time");


clear DELTATimetable;
clear BaseDataMatched;
clear BaseData;
clear BASEUniform;
clear ZTCFUniform;
clear DELTAUniform;
clear mdlWks;
clear Ts;

%Convert the time vector back to normal time rather than duration
BASETime=seconds(BASE.Time);
BASE.Time=BASETime;

DELTATime=seconds(DELTA.Time);
DELTA.Time=DELTATime;

ZTCFTime=seconds(ZTCF.Time);
ZTCF.Time=ZTCFTime;

BASEQ=BASE;
ZTCFQ=ZTCF;
DELTAQ=DELTA;

clear BASETime;
clear DELTATime;
clear ZTCFTime;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the correction program for linear work and linear impulse for ZTCF and
% DELTA.
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_UpdateCalcsforImpulseandWork;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Q spacing program for the plots:
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_QTableTimeChange;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Calculation for Total Work and Power at Each Joint
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_TotalWorkandPowerCalculation;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Club and Hand Path Vectors in the Tables
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_CHPandMPPCalculation;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Table of Values Script and Generate Data for Shaft Quivers at Times
% of interest
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_TableofValues;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the tables
cd(matlabdrive)
cd '2DModel'
mkdir 'Tables';
cd(matlabdrive);
cd '2DModel/Tables/';
save('BASE.mat',"BASE");
save('ZTCF.mat',"ZTCF");
save('DELTA.mat',"DELTA");
save('BASEQ.mat',"BASEQ");
save('ZTCFQ.mat',"ZTCFQ");
save('DELTAQ.mat',"DELTAQ");
save("ClubQuiverAlphaReversal.mat","ClubQuiverAlphaReversal");
save("ClubQuiverMaxCHS.mat","ClubQuiverMaxCHS");
save("ClubQuiverZTCFAlphaReversal.mat","ClubQuiverZTCFAlphaReversal");
save("ClubQuiverDELTAAlphaReversal.mat","ClubQuiverDELTAAlphaReversal");
save("SummaryTable.mat","SummaryTable");

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the ZVCF Script:
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_ZVCF_GENERATOR;

% Save the ZVCF Tables
cd(matlabdrive);
cd '2DModel';
mkdir 'Tables';
cd(matlabdrive);
cd '2DModel/Tables/';
save('ZVCFTable.mat',"ZVCFTable");
save('ZVCFTableQ.mat',"ZVCFTableQ");

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Plotting Script:
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_AllPlots;

clear ZTCFPercentComplete;

%Retun to 2DModel home page on matlab drive
cd(matlabdrive);
cd '2DModel';

clear ClubQuiverAlphaReversal;
clear ClubQuiverMaxCHS;
clear ClubQuiverZTCFAlphaReversal;
clear ClubQuiverDELTAAlphaReversal;
clear SummaryTable;

PlotStatus="Done Plotting"
clear PlotStatus;

