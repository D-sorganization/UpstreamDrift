%Generate Club, Shaft, and Hand Path
QuiverClubandShaftExcel

%Generate LH Force
LHForce= quiver3(Data{:,2},Data{:,3},Data{:,4},Data{:,29},Data{:,30},Data{:,31});
LHForce.LineWidth=1;
LHForce.Color=[0 0 1];
LHForce.AutoScaleFactor=1;
LHForce.MaxHeadSize=0.1;

%Generate RH Force
RHForce=quiver3(Data{:,5},Data{:,6},Data{:,7},Data{:,32},Data{:,33},Data{:,34});
RHForce.LineWidth=1;
RHForce.Color=[1 0 0];
RHForce.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHForce.AutoScaleFactor=LHForce.ScaleFactor/RHForce.ScaleFactor*LHForce.AutoScaleFactor;

%Add Legend to Plot
legend('','','','LHForce','RHForce')

%Add a Title
title('Total Force');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Figure 1 - Total Force LHRH');

%Close Figure
close(1);