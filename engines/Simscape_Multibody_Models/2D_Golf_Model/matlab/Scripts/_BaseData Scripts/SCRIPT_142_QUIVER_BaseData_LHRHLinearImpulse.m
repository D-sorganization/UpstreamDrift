%Generate Club Quiver Plot
figure(142);
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

%Add Legend to Plot
legend('','','','','LH Linear Impulse','RH Linear Impulse');

%Add a Title
title('LH and RH Linear Impulse on Club');
subtitle('BASE');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('BaseData Quiver Plots/BASE_Quiver Plot - LHRH Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(142);

%Clear Figure from Workspace
clear LHLinearImpulseQuiver;
clear RHLinearImpulseQuiver;
clear ZTCFLHLinearImpulseQuiver;
clear ZTCFRHLinearImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
