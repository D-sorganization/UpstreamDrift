%Computation of Clubhead Path Vectors

%Find height of table
H=height(BASEQ);

%Find number of times to iterate ommiting last row
h=H-1;

%Generate 
for i=1:h
j=i+1;

%Compute Components of Path Vectors
CHPxTemp=BASEQ.CHx(j,1)-BASEQ.CHx(i,1);
CHPyTemp=BASEQ.CHy(j,1)-BASEQ.CHy(i,1);
CHPzTemp=BASEQ.CHz(j,1)-BASEQ.CHz(i,1);
MPPxTemp=BASEQ.MPx(j,1)-BASEQ.MPx(i,1);
MPPyTemp=BASEQ.MPy(j,1)-BASEQ.MPy(i,1);
MPPzTemp=BASEQ.MPz(j,1)-BASEQ.MPz(i,1);

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

%Write to BASEQ Table
BASEQ.("CHPx")=CHPx;
BASEQ.("CHPy")=CHPy;
BASEQ.("CHPz")=CHPz;
BASEQ.("MPPx")=MPPx;
BASEQ.("MPPy")=MPPy;
BASEQ.("MPPz")=MPPz;

%Write to ZTCF Table
ZTCFQ.("CHPx")=CHPx;
ZTCFQ.("CHPy")=CHPy;
ZTCFQ.("CHPz")=CHPz;
ZTCFQ.("MPPx")=MPPx;
ZTCFQ.("MPPy")=MPPy;
ZTCFQ.("MPPz")=MPPz;

%Write to Delta Table
DELTAQ.("CHPx")=CHPx;
DELTAQ.("CHPy")=CHPy;
DELTAQ.("CHPz")=CHPz;
DELTAQ.("MPPx")=MPPx;
DELTAQ.("MPPy")=MPPy;
DELTAQ.("MPPz")=MPPz;

%Cleanup
clear CHPx;
clear CHPy;
clear CHPz;
clear MPPx;
clear MPPy;
clear MPPz;

