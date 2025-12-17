%SCRIPT_QTableTimeChange
%Turn off the warning that a directory already exists when you create it.
warning('off', 'MATLAB:MKDIR:DirectoryExists');

%Create the tables from nontruncated data:
BASEQ=BASE;
ZTCFQ=ZTCF;
DELTAQ=DELTA;

%Desired time step:
TsQ=0.0025;

%Generate duration times for each table:
BASETime=seconds(BASEQ.Time);
ZTCFTime=seconds(ZTCFQ.Time);
DELTATime=seconds(DELTAQ.Time);

%Create Versions of the Table to Modify
BASETemp=BASEQ;
ZTCFTemp=ZTCFQ;
DELTATemp=DELTAQ;

%Write duration times into the tables:
BASETemp.('t')=BASETime;
ZTCFTemp.('t')=ZTCFTime;
DELTATemp.('t')=DELTATime;

%Create a timetable using the duration timnes generated above:
BASETimetableTemp=table2timetable(BASETemp,"RowTimes","t");
ZTCFTimetableTemp=table2timetable(ZTCFTemp,"RowTimes","t");
DELTATimetableTemp=table2timetable(DELTATemp,"RowTimes","t");

%Remove the remaining time variable from the Timetable:
BASETimetable=removevars(BASETimetableTemp,'Time');
ZTCFTimetable=removevars(ZTCFTimetableTemp,'Time');
DELTATimetable=removevars(DELTATimetableTemp,'Time');


BASEQUniform=retime(BASETimetable,'regular','spline','TimeStep',seconds(TsQ));
ZTCFQUniform=retime(ZTCFTimetable,'regular','spline','TimeStep',seconds(TsQ));
DELTAQUniform=retime(DELTATimetable,'regular','spline','TimeStep',seconds(TsQ));

%Cleanup a bit
clear BASETimetable;
clear BASETemp;
clear BASETime;
clear BASETimetableTemp;
clear ZTCFTimetable
clear ZTCFTemp;
clear ZTCFTime;
clear ZTCFTimetableTemp;
clear DELTATimetable
clear DELTATemp;
clear DELTATime;
clear DELTATimetableTemp;

DELTAQ=timetable2table(DELTAQUniform,"ConvertRowTimes",true);
DELTAQ=renamevars(DELTAQ,"t","Time");
BASEQ=timetable2table(BASEQUniform,"ConvertRowTimes",true);
BASEQ=renamevars(BASEQ,"t","Time");
ZTCFQ=timetable2table(ZTCFQUniform,"ConvertRowTimes",true);
ZTCFQ=renamevars(ZTCFQ,"t","Time");

%Cleanup
clear DELTATimetable;
clear BASETimetable;
clear ZTCFTimetable;
clear BASEQUniform;
clear ZTCFQUniform;
clear DELTAQUniform;
clear TsQ;

%Convert the time vector back to normal time rather than duration
BASEQTime=seconds(BASEQ.Time);
BASEQ.Time=BASEQTime;
DELTAQTime=seconds(DELTAQ.Time);
DELTAQ.Time=DELTAQTime;
ZTCFQTime=seconds(ZTCFQ.Time);
ZTCFQ.Time=ZTCFQTime;

clear BASEQTime;
clear DELTAQTime;
clear ZTCFQTime;

cd(matlabdrive);
mkdir 'Tables';
cd 'Tables/';
save('BASEQ.mat',"BASEQ");
save('ZTCFQ.mat',"ZTCFQ");
save('DELTAQ.mat',"DELTAQ");