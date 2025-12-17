%Generate Club, Shaft, and Hand Path
QuiverClubandShaftExcel

%Generate ZTCF LH Force
LHZTCF= quiver3(Data{:,2},Data{:,3},Data{:,4},Data{:,35},Data{:,36},Data{:,37});
LHZTCF.LineWidth=1;
LHZTCF.Color=[0 0.5 1];
LHZTCF.AutoScaleFactor=1;
LHZTCF.MaxHeadSize=0.1;

%Generate ZTCF RH Force
RHZTCF=quiver3(Data{:,5},Data{:,6},Data{:,7},Data{:,38},Data{:,39},Data{:,40});
RHZTCF.LineWidth=1;
RHZTCF.Color=[1 0.5 0];
RHZTCF.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHZTCF.AutoScaleFactor=LHZTCF.ScaleFactor/RHZTCF.ScaleFactor*LHZTCF.AutoScaleFactor;

%Add Legend to Plot
legend('','','','LH ZTCF Force','RH ZTCF Force')

%Add a Title
title('ZTCF Force');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Figure 2 - ZTCF Force LHRH');

%Close Figure
%close(1);