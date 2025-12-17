%Generate Club Quiver Plot
figure(914);
run SCRIPT_QuiverClubandShaftData.m;

%Generate LW Total Force Quiver Plot
LHForceQuiver=quiver3(Data.LWx(:,1),Data.LWy(:,1),Data.LWz(:,1),Data.LWonClubFGlobal(:,1),Data.LWonClubFGlobal(:,2),Data.LWonClubFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(Data.RWx(:,1),Data.RWy(:,1),Data.RWz(:,1),Data.RWonClubFGlobal(:,1),Data.RWonClubFGlobal(:,2),Data.RWonClubFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[1 0 0];
RHForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/RHForceQuiver.ScaleFactor;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),Data.TotalHandForceGlobal(:,1),Data.TotalHandForceGlobal(:,2),Data.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
NetForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/NetForceQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH Force','RH Force','Net Force');

%Add a Title
title('Total Force');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Hand Forces');
pause(PauseTime);

%Close Figure
close(914);

%Clear Figure from Workspace
clear LHForceQuiver;
clear RHForceQuiver;
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;