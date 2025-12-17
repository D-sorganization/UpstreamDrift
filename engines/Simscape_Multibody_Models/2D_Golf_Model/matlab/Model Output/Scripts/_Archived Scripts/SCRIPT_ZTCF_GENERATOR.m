% ZTCF Data Generation Script
% If you get errors relating to out.tout not being there it is because the
% table generation script is still turned on as a callback in the model
% itself. This ends up clearing the out variable before it can be used.
% Consider getting rid of the part where out is deleted or moving it to the
% end.

cd 'C:\Users\diete\MATLAB Drive';
%load("GolfSwing.slx");

% mdlWks Generation File Creation
mdlWks=get_param('GolfSwing','ModelWorkspace');

%Set the killswitch time to 1 second so it doesn't ever trigger
assignin(mdlWks,'KillswitchStepTime',1);

% Set up the model to return a single matlab object as output.
set_param(GolfSwing,"ReturnWorkspaceOutputs","on");
set_param(GolfSwing,"FastRestart","on");
set_param(GolfSwing,"MaxStep","0.001");

% Run the model to generate BaseData table. Save model output as "out".
out=sim(GolfSwing);

% Run Table Generation Script on "out"
SCRIPT_TableGeneration;

% Create table called BaseData with the "Data" Table from run with no
% killswitch.
BaseData=Data;

% Copy Data to ZTCF Table to Get Variable Names
ZTCFTable=Data; %Create a table called ZTCFTable from Data.
ZTCFTable(:,:)=[]; %Delete All Data in ZTCF Table and Replace with Blanks

% Begin Generation of ZTCF Data by Looping
% stop=getVariable(mdlWks,'StopTime');(Trying to automate the number n here
% to see how many iterations to run).
% n=stop*100;

    for i=0:280
    %parfor i=0:280
 
    %Scale counter to match desired times
    j=i/1000;
    
    %Write step time to model workspace
    assignin(mdlWks,'KillswitchStepTime',j);     
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
clear mdlWks;
clear ZTCF;
clear ZTCFData;
"Simulation Complete";
clear ans;
ZTCF=ZTCFTable;
clear ZTCFTable;


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
TsQ=0.005;

BASEUniform=retime(BaseDataMatched,'regular','spline','TimeStep',seconds(Ts));
ZTCFUniform=retime(ZTCFTimetable,'regular','spline','TimeStep',seconds(Ts));
DELTAUniform=retime(DELTATimetable,'regular','spline','TimeStep',seconds(Ts));
BASEQUniform=retime(BaseDataMatched,'regular','spline','TimeStep',seconds(TsQ));
ZTCFQUniform=retime(ZTCFTimetable,'regular','spline','TimeStep',seconds(TsQ));
DELTAQUniform=retime(DELTATimetable,'regular','spline','TimeStep',seconds(TsQ));

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
DELTAQ=timetable2table(DELTAQUniform,"ConvertRowTimes",true);
DELTAQ=renamevars(DELTAQ,"t","Time");

BASE=timetable2table(BASEUniform,"ConvertRowTimes",true);
BASE=renamevars(BASE,"t","Time");
BASEQ=timetable2table(BASEQUniform,"ConvertRowTimes",true);
BASEQ=renamevars(BASEQ,"t","Time");

ZTCF=timetable2table(ZTCFUniform,"ConvertRowTimes",true);
ZTCF=renamevars(ZTCF,"t","Time");
ZTCFQ=timetable2table(ZTCFQUniform,"ConvertRowTimes",true);
ZTCFQ=renamevars(ZTCFQ,"t","Time");

clear DELTATimetable;
clear BaseDataMatched;
clear BaseData;
clear BASEUniform;
clear ZTCFUniform;
clear DELTAUniform;
clear DELTAQUniform;
clear ZTCFQUniform;
clear BASEQUniform;
clear Ts;
clear TsQ;

%Convert the time vector back to normal time rather than duration
BASETime=seconds(BASE.Time);
BASE.Time=BASETime;
BASEQTime=seconds(BASEQ.Time);
BASEQ.Time=BASEQTime;

DELTATime=seconds(DELTA.Time);
DELTA.Time=DELTATime;
DELTAQTime=seconds(DELTAQ.Time);
DELTAQ.Time=DELTAQTime;

ZTCFTime=seconds(ZTCF.Time);
ZTCF.Time=ZTCFTime;
ZTCFQTime=seconds(ZTCFQ.Time);
ZTCFQ.Time=ZTCFQTime;

clear BASETime;
clear DELTATime;
clear ZTCFTime;
clear BASEQTime;
clear DELTAQTime;
clear ZTCFQTime;

save('BASE.mat',"BASE");
save('ZTCF.mat',"ZTCF");
save('DELTA.mat',"DELTA");
save('BASEQ.mat',"BASEQ");
save('ZTCFQ.mat',"ZTCFQ");
save('DELTAQ.mat',"DELTAQ");

