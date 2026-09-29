%Generate Club Quiver Plot
figure(724);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Linear Impulse on Club Quiver Plot
TotalLinearImpulse=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.LinearImpulseonClub(:,1),BASEQ.LinearImpulseonClub(:,2),BASEQ.LinearImpulseonClub(:,3));
TotalLinearImpulse.LineWidth=1;
TotalLinearImpulse.Color=[0 1 0];
TotalLinearImpulse.MaxHeadSize=0.1;
TotalLinearImpulse.AutoScaleFactor=3;

%Generate ZTCF Total Linear Impulse on Club Quiver Plot
ZTCFTotalLinearImpulse=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.LinearImpulseonClub(:,1),ZTCFQ.LinearImpulseonClub(:,2),ZTCFQ.LinearImpulseonClub(:,3));
ZTCFTotalLinearImpulse.LineWidth=1;
ZTCFTotalLinearImpulse.Color=[0 0.3 0];
ZTCFTotalLinearImpulse.MaxHeadSize=0.1;
%Correct scaling on Linear Impulses so that ZTCF and BASE are scaled the same.
ZTCFTotalLinearImpulse.AutoScaleFactor=TotalLinearImpulse.ScaleFactor/ZTCFTotalLinearImpulse.ScaleFactor;

%Add Legend to Plot
legend('','','','','BASE - Total Linear Impulse','ZTCF - Total Linear Impulse');

%Add a Title
title('Linear Impulse on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Total Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(724);

%Clear Figure from Workspace
clear TotalLinearImpulse;
clear MPSumofMomentsQuiver;
clear ZTCFTotalLinearImpulse;
clear ZTCFMPSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
