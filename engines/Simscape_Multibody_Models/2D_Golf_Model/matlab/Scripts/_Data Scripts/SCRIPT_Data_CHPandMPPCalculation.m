%Computation of Clubhead Path Vectors

%Find height of table
H=height(Data);

%Find number of times to iterate ommiting last row
h=H-1;

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

%Write to Data Table
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

