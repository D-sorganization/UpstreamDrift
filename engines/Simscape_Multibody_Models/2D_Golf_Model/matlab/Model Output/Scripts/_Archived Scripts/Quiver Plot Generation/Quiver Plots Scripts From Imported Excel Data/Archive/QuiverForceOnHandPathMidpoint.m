%Generate Club, Shaft, and Hand Path
QuiverClubandShaftExcel

%Generate Force on Hand Path
FHPMP= quiver3(Data{:,8},Data{:,9},Data{:,10},Data{:,44},Data{:,45},Data{:,46});
FHPMP.LineWidth=1;
FHPMP.Color=[0 0.5 1];
FHPMP.AutoScaleFactor=2.8;
FHPMP.MaxHeadSize=0.1;

%Add Legend to Plot
legend('','','','Force on Hand Path')

%Add a Title
title('Force on Handpath');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Figure 4 - Total Force on Handpath');

%Close Figure
%close(1);