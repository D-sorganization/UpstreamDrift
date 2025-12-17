%Generate Club, Shaft, and Hand Path
QuiverClubandShaftExcel

%Generate Total Force on Midpoint
TotalForce= quiver3(Data{:,8},Data{:,9},Data{:,10},Data{:,23},Data{:,24},Data{:,25});
TotalForce.LineWidth=1;
TotalForce.Color=[0 0 1];
TotalForce.AutoScaleFactor=1;
TotalForce.MaxHeadSize=0.1;

%Generate ZTCF Force on Midpoint
ZTCFForce=quiver3(Data{:,8},Data{:,9},Data{:,10},Data{:,41},Data{:,42},Data{:,43});
ZTCFForce.LineWidth=1;
ZTCFForce.Color=[1 0 0];
ZTCFForce.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
ZTCFForce.AutoScaleFactor=TotalForce.ScaleFactor/ZTCFForce.ScaleFactor*TotalForce.AutoScaleFactor;

%Generate Delta Force on Midpoint
DeltaForce=quiver3(Data{:,8},Data{:,9},Data{:,10},Data{:,71},Data{:,72},Data{:,73});
DeltaForce.LineWidth=1;
DeltaForce.Color=[0 1 0];
%Correct scaling so that LH and RH are scaled the same.
DeltaForce.AutoScaleFactor=TotalForce.ScaleFactor/DeltaForce.ScaleFactor*TotalForce.AutoScaleFactor;

%Add Legend to Plot
legend('','','','Total Force','ZTCF Force','Delta Force')

%Add a Title
title('Total Force on Midpoint','Total, ZTCF, Delta');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Figure 5 - Total Force on Midpoint Comparison');

%Close Figure
%close(1);