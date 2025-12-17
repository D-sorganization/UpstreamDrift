%Generate Club Quiver Plot
figure(717);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Force Quiver Plot
LHForceQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWonClubFGlobal(:,1),BASEQ.LWonClubFGlobal(:,2),BASEQ.LWonClubFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 1 0];
LHForceQuiver.MaxHeadSize=0.1;
LHForceQuiver.AutoScaleFactor=3;

%Generate Right Hand Force Quiver Plot
RHForceQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RWonClubFGlobal(:,1),BASEQ.RWonClubFGlobal(:,2),BASEQ.RWonClubFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[.8 .2 0];
RHForceQuiver.MaxHeadSize=0.1;
%Correct scaling on Forces so that ZTCF and BASE are scaled the same.
RHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/RHForceQuiver.ScaleFactor;

%Generate ZTCF Left Hand Force Quiver Plot
ZTCFLHForceQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LWonClubFGlobal(:,1),ZTCFQ.LWonClubFGlobal(:,2),ZTCFQ.LWonClubFGlobal(:,3));
ZTCFLHForceQuiver.LineWidth=1;
ZTCFLHForceQuiver.Color=[0 0.3 0];
ZTCFLHForceQuiver.MaxHeadSize=0.1;
%Correct scaling on Forces so that ZTCF and BASE are scaled the same.
ZTCFLHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/ZTCFLHForceQuiver.ScaleFactor;

%Generate ZTCF Right Hand Force Quiver Plot
ZTCFRHForceQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RWonClubFGlobal(:,1),ZTCFQ.RWonClubFGlobal(:,2),ZTCFQ.RWonClubFGlobal(:,3));
ZTCFRHForceQuiver.LineWidth=1;
ZTCFRHForceQuiver.Color=[.5 .5 0];
ZTCFRHForceQuiver.MaxHeadSize=0.1;
%Correct scaling on Equivalent Couple so that ZTCF and BASE are scaled the same.
ZTCFRHForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/ZTCFRHForceQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','BASE - LH Force','BASE - RH Force','ZTCF - LH Force','ZTCF - RH Force');

%Add a Title
title('LH and RH Force on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - LHRH Force on Club');
pause(PauseTime);

%Close Figure
close(717);

%Clear Figure from Workspace
clear LHForceQuiver;
clear RHForceQuiver;
clear ZTCFLHForceQuiver;
clear ZTCFRHForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
