%Generate a table with a time column and the variable name set to time.
Time=out.tout;
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

%Clean Up Workspace After Running and Generating the Table
clear i;
clear signalName;
clear signalData;
clear Time;

%Generate Shaft and Grip Vector Components for Quivers Plot Use

%Grip Scale Factor (Size up grip vector for graphics)
GripScale=1.5;

%Generate Grip Vector in Table
Data.Gripdx=GripScale.*(Data.RWx-Data.Buttx);
Data.Gripdy=GripScale.*(Data.RWy-Data.Butty);
Data.Gripdz=GripScale.*(Data.RWz-Data.Buttz);
clear GripScale;

%Generate Shaft Vector in Table
Data.Shaftdx=Data.CHx-Data.RWx;
Data.Shaftdy=Data.CHy-Data.RWy;
Data.Shaftdz=Data.CHz-Data.RWz;

%Generate Left Forearm Vector in Table
Data.LeftForearmdx=Data.LWx-Data.LEx;
Data.LeftForearmdy=Data.LWy-Data.LEy;
Data.LeftForearmdz=Data.LWz-Data.LEz;

%Generate Left Forearm Vector in Table
Data.RightForearmdx=Data.RWx-Data.REx;
Data.RightForearmdy=Data.RWy-Data.REy;
Data.RightForearmdz=Data.RWz-Data.REz;

%Generate Left Upper Arm Vector in Table
Data.LeftArmdx=Data.LEx-Data.LSx;
Data.LeftArmdy=Data.LEy-Data.LSy;
Data.LeftArmdz=Data.LEz-Data.LSz;

%Generate Right Upper Arm Vector in Table
Data.RightArmdx=Data.REx-Data.RSx;
Data.RightArmdy=Data.REy-Data.RSy;
Data.RightArmdz=Data.REz-Data.RSz;

%Generate Left Shoulder Vector
Data.LeftShoulderdx=Data.LSx-Data.HUBx;
Data.LeftShoulderdy=Data.LSy-Data.HUBy;
Data.LeftShoulderdz=Data.LSz-Data.HUBz;

%Generate Right Shoulder Vector
Data.RightShoulderdx=Data.RSx-Data.HUBx;
Data.RightShoulderdy=Data.RSy-Data.HUBy;
Data.RightShoulderdz=Data.RSz-Data.HUBz;

