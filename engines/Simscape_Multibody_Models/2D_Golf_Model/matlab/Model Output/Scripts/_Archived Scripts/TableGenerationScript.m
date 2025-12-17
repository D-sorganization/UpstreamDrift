%Generate a table with a time column and the variable name set to time.
Time=out.tout
Data = table(Time,'VariableNames', {'Time'});
%Loop through each dataset element to add it to the table
for i=1:out.logsout.numElements;
    %Get signal name
    signalName=out.logsout.getElement(i).Name;
    %Get signal data
    signalData=out.logsout.getElement(i).Values.Data;
    %Add the data as a new column in the table
    Data.(signalName)=signalData;
end

clear i;

%Generate Shaft and Grip Vector Components for Quivers Plot Use

%Generate Grip Vector in Table
Data.Gripdx=Data.RWx-Data.Buttx;
Data.Gripdy=Data.RWy-Data.Butty;
Data.Gripdz=Data.RWz-Data.Buttz;

%Generate Shaft Vector in Table
Data.Shaftdx=Data.CHx-Data.RWx;
Data.Shaftdy=Data.CHy-Data.RWy;
Data.Shaftdz=Data.CHz-Data.RWz;
 
% %Generate Hand Path Vector
% counter=height(Data)-1;
% 
% for i= i=1:counter;
%     HandPathdx=(Data.MidHandPosition(i+1,1)-Data.MidHandPosition(i,1))
%     HandPathdy=(Data.MidHandPosition(i+1,2)-Data.MidHandPosition(i,2))
%     HandPathdz=(Data.MidHandPosition(i+1,3)-Data.MidHandPosition(i,3))
% end
% 
% %Add Hand Path Vector Data to Table
