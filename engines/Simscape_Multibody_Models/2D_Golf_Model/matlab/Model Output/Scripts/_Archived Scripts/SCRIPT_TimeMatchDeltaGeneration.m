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

%Cleanup a bit
clear BaseDataTimetable;
clear BaseDataTemp;
clear BaseDataTime;
clear BaseDataTimetableTemp;
clear ZTCFTimetable
clear ZTCFTemp;
clear ZTCFTime;
clear ZTCFTimetableTemp;

%Convert Delta to Table:
DELTA=timetable2table(DELTATimetable,"ConvertRowTimes",true);
DELTA=renamevars(DELTA,"t","Time");
BASE=timetable2table(BaseDataMatched,"ConvertRowTimes",true);
BASE=renamevars(BASE,"t","Time");

clear DELTATimetable;
clear BaseDataMatched;
clear BaseData;

BASETime=seconds(BASE.Time);
BASE.Time=BASETime;

clear BASETime;



