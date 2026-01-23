%Generate Club Quiver Plot
figure(314);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LW Total Force Quiver Plot
LHForceQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LWonClubFGlobal(:,1),ZTCFQ.LWonClubFGlobal(:,2),ZTCFQ.LWonClubFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RWonClubFGlobal(:,1),ZTCFQ.RWonClubFGlobal(:,2),ZTCFQ.RWonClubFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[1 0 0];
RHForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/RHForceQuiver.ScaleFactor;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.TotalHandForceGlobal(:,1),ZTCFQ.TotalHandForceGlobal(:,2),ZTCFQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
NetForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/NetForceQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH Force','RH Force','Net Force');

%Add a Title
title('Total Force');
subtitle('ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Hand Forces');
pause(PauseTime);

%Close Figure
close(314);

%Clear Figure from Workspace
clear LHForceQuiver;
clear RHForceQuiver;
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;