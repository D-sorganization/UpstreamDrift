%Generate Club Quiver Plot
figure(814);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LW Total Force Quiver Plot
LHForceQuiver=quiver3(ZVCFTableQ.LWx(:,1),ZVCFTableQ.LWy(:,1),ZVCFTableQ.LWz(:,1),ZVCFTableQ.LWonClubFGlobal(:,1),ZVCFTableQ.LWonClubFGlobal(:,2),ZVCFTableQ.LWonClubFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(ZVCFTableQ.RWx(:,1),ZVCFTableQ.RWy(:,1),ZVCFTableQ.RWz(:,1),ZVCFTableQ.RWonClubFGlobal(:,1),ZVCFTableQ.RWonClubFGlobal(:,2),ZVCFTableQ.RWonClubFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[1 0 0];
RHForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
RHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/RHForceQuiver.ScaleFactor;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(ZVCFTableQ.MPx(:,1),ZVCFTableQ.MPy(:,1),ZVCFTableQ.MPz(:,1),ZVCFTableQ.TotalHandForceGlobal(:,1),ZVCFTableQ.TotalHandForceGlobal(:,2),ZVCFTableQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
%Correct scaling so that LH and RH are scaled the same.
NetForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/NetForceQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH Force','RH Force','Net Force');

%Add a Title
title('Hand Forces');
subtitle('ZVCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZVCF Quiver Plots/ZVCF_Quiver Plot - Hand Forces');
%PauseTime=1;
pause(PauseTime);

%Close Figure
close(814);

%Clear Figure from Workspace
clear LHForceQuiver;
clear RHForceQuiver;
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
clear PauseTime;
