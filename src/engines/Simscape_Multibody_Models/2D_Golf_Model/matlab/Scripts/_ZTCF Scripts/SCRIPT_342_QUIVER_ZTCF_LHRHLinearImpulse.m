%Generate Club Quiver Plot
figure(342);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Linear Impulse Quiver Plot
LHLinearImpulseQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LHLinearImpulseonClub(:,1),ZTCFQ.LHLinearImpulseonClub(:,2),ZTCFQ.LHLinearImpulseonClub(:,3));
LHLinearImpulseQuiver.LineWidth=1;
LHLinearImpulseQuiver.Color=[0 1 0];
LHLinearImpulseQuiver.MaxHeadSize=0.1;
LHLinearImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Linear Impulse Quiver Plot
RHLinearImpulseQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RHLinearImpulseonClub(:,1),ZTCFQ.RHLinearImpulseonClub(:,2),ZTCFQ.RHLinearImpulseonClub(:,3));
RHLinearImpulseQuiver.LineWidth=1;
RHLinearImpulseQuiver.Color=[.8 .2 0];
RHLinearImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Linear Impulses so that LH and RH are scaled the same.
RHLinearImpulseQuiver.AutoScaleFactor=LHLinearImpulseQuiver.ScaleFactor/RHLinearImpulseQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH Linear Impulse','RH Linear Impulse');

%Add a Title
title('LH and RH Linear Impulse on Club');
subtitle('ZTCF');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - LHRH Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(342);

%Clear Figure from Workspace
clear LHLinearImpulseQuiver;
clear RHLinearImpulseQuiver;
clear ZTCFLHLinearImpulseQuiver;
clear ZTCFRHLinearImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
