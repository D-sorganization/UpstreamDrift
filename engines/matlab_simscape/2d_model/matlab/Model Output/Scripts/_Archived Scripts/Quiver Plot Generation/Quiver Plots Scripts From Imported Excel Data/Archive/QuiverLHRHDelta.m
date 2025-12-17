%Reads from a Table named Data in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate Club and Shaft Plot
QuiverClubandShaftExcel;

%Generate LH Delta
LHDelta=quiver3(Data{:,2},Data{:,3},Data{:,4},Data{:,11},Data{:,12},Data{:,13});
LHDelta.LineWidth=1;
LHDelta.Color=[0.89 0.66 0.13];
LHDelta.AutoScaleFactor=1;
LHDelta.MaxHeadSize=0.1;

%Generate RH Delta
RHDelta=quiver3(Data{:,5},Data{:,6},Data{:,7},Data{:,14},Data{:,15},Data{:,16});
RHDelta.LineWidth=1;
RHDelta.Color=[0.13 0.55 0.89];
RHDelta.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHDelta.AutoScaleFactor=LHDelta.ScaleFactor/RHDelta.ScaleFactor*LHDelta.AutoScaleFactor;

%Add Legend to Plot
legend('','','','LH Delta Force','RH Delta Force')

%Add a Title
title('Delta Force','Total Minus ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Figure 3 - Delta Force LHRH');

%Close Figure
%close(1);