%Generate Club Quiver Plot
figure(718);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Linear Impulse Quiver Plot
LHLinearImpulseQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LHLinearImpulseonClub(:,1),BASEQ.LHLinearImpulseonClub(:,2),BASEQ.LHLinearImpulseonClub(:,3));
LHLinearImpulseQuiver.LineWidth=1;
LHLinearImpulseQuiver.Color=[0 1 0];
LHLinearImpulseQuiver.MaxHeadSize=0.1;
LHLinearImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Linear Impulse Quiver Plot
RHLinearImpulseQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RHLinearImpulseonClub(:,1),BASEQ.RHLinearImpulseonClub(:,2),BASEQ.RHLinearImpulseonClub(:,3));
RHLinearImpulseQuiver.LineWidth=1;
RHLinearImpulseQuiver.Color=[.8 .2 0];
RHLinearImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Linear Impulses so that ZTCF and BASE are scaled the same.
RHLinearImpulseQuiver.AutoScaleFactor=LHLinearImpulseQuiver.ScaleFactor/RHLinearImpulseQuiver.ScaleFactor;

%Generate ZTCF Left Hand Linear Impulse Quiver Plot
ZTCFLHLinearImpulseQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LHLinearImpulseonClub(:,1),ZTCFQ.LHLinearImpulseonClub(:,2),ZTCFQ.LHLinearImpulseonClub(:,3));
ZTCFLHLinearImpulseQuiver.LineWidth=1;
ZTCFLHLinearImpulseQuiver.Color=[0 0.3 0];
ZTCFLHLinearImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Linear Impulses so that ZTCF and BASE are scaled the same.
ZTCFLHLinearImpulseQuiver.AutoScaleFactor=LHLinearImpulseQuiver.ScaleFactor/ZTCFLHLinearImpulseQuiver.ScaleFactor;

%Generate ZTCF Right Hand Linear Impulse Quiver Plot
ZTCFRHLinearImpulseQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RHLinearImpulseonClub(:,1),ZTCFQ.RHLinearImpulseonClub(:,2),ZTCFQ.RHLinearImpulseonClub(:,3));
ZTCFRHLinearImpulseQuiver.LineWidth=1;
ZTCFRHLinearImpulseQuiver.Color=[.5 .5 0];
ZTCFRHLinearImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Linear Impulse so that ZTCF and BASE are scaled the same.
ZTCFRHLinearImpulseQuiver.AutoScaleFactor=LHLinearImpulseQuiver.ScaleFactor/ZTCFRHLinearImpulseQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','BASE - LH Linear Impulse','BASE - RH Linear Impulse','ZTCF - LH Linear Impulse','ZTCF - RH Linear Impulse');

%Add a Title
title('LH and RH Linear Impulse on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - LHRH Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(718);

%Clear Figure from Workspace
clear LHLinearImpulseQuiver;
clear RHLinearImpulseQuiver;
clear ZTCFLHLinearImpulseQuiver;
clear ZTCFRHLinearImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
