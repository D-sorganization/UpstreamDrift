%Generate Club Quiver Plot
figure(3);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LW Total Force Quiver Plot
LHForceQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWonClubFGlobal(:,1),BASEQ.LWonClubFGlobal(:,2),BASEQ.LWonClubFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RWonClubFGlobal(:,1),BASEQ.RWonClubFGlobal(:,2),BASEQ.RWonClubFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[1 0 0];
RHForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/RHForceQuiver.ScaleFactor;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalHandForceGlobal(:,1),BASEQ.TotalHandForceGlobal(:,2),BASEQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
NetForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/NetForceQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','LH Force','RH Force','Net Force');

%Add a Title
title('Total Force');
subtitle('COMPARISON');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Hand Forces');

%Close Figure
close(3);

%Clear Figure from Workspace
clear LHForceQuiver;
clear RHForceQuiver;
clear NetForceQuiver;
clear Grip;
clear Shaft;