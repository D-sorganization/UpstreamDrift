%SCRIPT_QTableTimeChange
%Turn off the warning that a directory already exists when you create it.
warning('off', 'MATLAB:MKDIR:DirectoryExists');

%Create the tables from nontruncated data:
ZVCFTableQ=ZVCFTable;

%Desired time step:
TsQ=0.0025;

%Generate duration times for each table:
ZVCFTableTime=seconds(ZVCFTableQ.Time);

%Create Versions of the Table to Modify
ZVCFTableTemp=ZVCFTableQ;

%Write duration times into the tables:
ZVCFTableTemp.('t')=ZVCFTableTime;

%Create a timetable using the duration timnes generated above:
ZVCFTableTimetableTemp=table2timetable(ZVCFTableTemp,"RowTimes","t");

%Remove the remaining time variable from the Timetable:
ZVCFTableTimetable=removevars(ZVCFTableTimetableTemp,'Time');

ZVCFTableQUniform=retime(ZVCFTableTimetable,'regular','spline','TimeStep',seconds(TsQ));

%Cleanup a bit
clear ZVCFTableTimetable;
clear ZVCFTableTemp;
clear ZVCFTableTime;
clear ZVCFTableTimetableTemp;

ZVCFTableQ=timetable2table(ZVCFTableQUniform,"ConvertRowTimes",true);
ZVCFTableQ=renamevars(ZVCFTableQ,"t","Time");

%Cleanup
clear ZVCFTableTimetable;
clear ZVCFTableQUniform;
clear TsQ;

%Convert the time vector back to normal time rather than duration
ZVCFTableQTime=seconds(ZVCFTableQ.Time);
ZVCFTableQ.Time=ZVCFTableQTime;

clear ZVCFTableQTime;

cd(matlabdrive);
cd '2DModel';
mkdir 'Tables';
cd 'Tables/';
save('ZVCFTableQ.mat',"ZVCFTableQ");
