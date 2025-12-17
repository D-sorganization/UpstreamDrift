%Generate Club Quiver Plot
figure(751);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Moment of Force Quiver Plot
LWSumofMomentsQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LHMOFonClubGlobal(:,1),BASEQ.LHMOFonClubGlobal(:,2),BASEQ.LHMOFonClubGlobal(:,3));
LWSumofMomentsQuiver.LineWidth=1;
LWSumofMomentsQuiver.Color=[0 1 0];
LWSumofMomentsQuiver.MaxHeadSize=0.1;
LWSumofMomentsQuiver.AutoScaleFactor=3;

%Generate Right Hand Moment of Force Quiver Plot
RWSumofMomentsQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RHMOFonClubGlobal(:,1),BASEQ.RHMOFonClubGlobal(:,2),BASEQ.RHMOFonClubGlobal(:,3));
RWSumofMomentsQuiver.LineWidth=1;
RWSumofMomentsQuiver.Color=[.8 .2 0];
RWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that ZTCF and BASE are scaled the same.
RWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/RWSumofMomentsQuiver.ScaleFactor;

%Generate ZTCF Left Hand Moment of Force Quiver Plot
ZTCFLWSumofMomentsQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LHMOFonClubGlobal(:,1),ZTCFQ.LHMOFonClubGlobal(:,2),ZTCFQ.LHMOFonClubGlobal(:,3));
ZTCFLWSumofMomentsQuiver.LineWidth=1;
ZTCFLWSumofMomentsQuiver.Color=[0 0.3 0];
ZTCFLWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that BASE, ZTCF, and DELTA are scaled the same.
ZTCFLWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/ZTCFLWSumofMomentsQuiver.ScaleFactor;

%Generate ZTCF Right Hand Moment of Force Quiver Plot
ZTCFRWSumofMomentsQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RHMOFonClubGlobal(:,1),ZTCFQ.RHMOFonClubGlobal(:,2),ZTCFQ.RHMOFonClubGlobal(:,3));
ZTCFRWSumofMomentsQuiver.LineWidth=1;
ZTCFRWSumofMomentsQuiver.Color=[.5 .5 0];
ZTCFRWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that BASE, ZTCF, and DELTA are scaled the same.
ZTCFRWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/ZTCFRWSumofMomentsQuiver.ScaleFactor;

%Generate DELTA Left Hand Moment of Force Quiver Plot
DELTALWSumofMomentsQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),DELTAQ.LHMOFonClubGlobal(:,1),DELTAQ.LHMOFonClubGlobal(:,2),DELTAQ.LHMOFonClubGlobal(:,3));
DELTALWSumofMomentsQuiver.LineWidth=1;
DELTALWSumofMomentsQuiver.Color=[0 1 1];
DELTALWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that BASE, ZTCF, and DELTA are scaled the same.
DELTALWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/DELTALWSumofMomentsQuiver.ScaleFactor;

%Generate DELTA Right Hand Moment of Force Quiver Plot
DELTARWSumofMomentsQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),DELTAQ.RHMOFonClubGlobal(:,1),DELTAQ.RHMOFonClubGlobal(:,2),DELTAQ.RHMOFonClubGlobal(:,3));
DELTARWSumofMomentsQuiver.LineWidth=1;
DELTARWSumofMomentsQuiver.Color=[0 .2 .8];
DELTARWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that BASE, ZTCF, and DELTA are scaled the same.
DELTARWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/DELTARWSumofMomentsQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','BASE - LW Moment of Force','BASE - RW Moment of Force','ZTCF - LW Moment of Force','ZTCF - RW Moment of Force','DELTA - LW Moment of Force','DELTA - RW Moment of Force');

%Add a Title
title('LW and RW Moment of Force BASE, ZTCF, and DELTA');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - LWRW Moment of Force BASE vs ZTCF vs DELTA');
pause(PauseTime);

%Close Figure
close(751);

%Clear Figure from Workspace
clear LWSumofMomentsQuiver;
clear DELTALWSumofMomentsQuiver;
clear DELTARWSumofMomentsQuiver;
clear RWSumofMomentsQuiver;
clear ZTCFLWSumofMomentsQuiver;
clear ZTCFRWSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
