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

%Generate Grip Vector in Table
Data.Gripdx=Data.RWx-Data.Buttx;
Data.Gripdy=Data.RWy-Data.Butty;
Data.Gripdz=Data.RWz-Data.Buttz;

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

%Computation of Clubhead Path Vectors

%Find height of table
H=height(Data);

%Find number of times to iterate ommiting last row
h=H-1

%Generate 
for i=1:h
j=i+1;

%Compute Components of Path Vectors
CHPxTemp=Data.CHx(j,1)-Data.CHx(i,1);
CHPyTemp=Data.CHy(j,1)-Data.CHy(i,1);
CHPzTemp=Data.CHz(j,1)-Data.CHz(i,1);
MPPxTemp=Data.MPx(j,1)-Data.MPx(i,1);
MPPyTemp=Data.MPy(j,1)-Data.MPy(i,1);
MPPzTemp=Data.MPz(j,1)-Data.MPz(i,1);

%Write the Components into an Array
CHPx(i,1)=CHPxTemp;
CHPy(i,1)=CHPyTemp;
CHPz(i,1)=CHPzTemp;
MPPx(i,1)=MPPxTemp;
MPPy(i,1)=MPPyTemp;
MPPz(i,1)=MPPzTemp;

end

%Cleanup
clear CHPxTemp;
clear CHPyTemp;
clear CHPzTemp;
clear MPPxTemp;
clear MPPyTemp;
clear MPPzTemp;
clear i;
clear j;

%Write Last Row Values into the Tables
%Generate Temp Files
CHPxTemp=CHPx(h,1);
CHPyTemp=CHPy(h,1);
CHPzTemp=CHPz(h,1);
MPPxTemp=MPPx(h,1);
MPPyTemp=MPPy(h,1);
MPPzTemp=MPPz(h,1);
%Write temp files to end of array
CHPx(H,1)=CHPxTemp;
CHPy(H,1)=CHPyTemp;
CHPz(H,1)=CHPzTemp;
MPPx(H,1)=MPPxTemp;
MPPy(H,1)=MPPyTemp;
MPPz(H,1)=MPPzTemp;

%Cleanup
clear H;
clear h;
clear CHPxTemp;
clear CHPyTemp;
clear CHPzTemp;
clear MPPxTemp;
clear MPPyTemp;
clear MPPzTemp;

%Write to Data File
Data.("CHPx")=CHPx;
Data.("CHPy")=CHPy;
Data.("CHPz")=CHPz;
Data.("MPPx")=MPPx;
Data.("MPPy")=MPPy;
Data.("MPPz")=MPPz;

clear CHPx;
clear CHPy;
clear CHPz;
clear MPPx;
clear MPPy;
clear MPPz;

