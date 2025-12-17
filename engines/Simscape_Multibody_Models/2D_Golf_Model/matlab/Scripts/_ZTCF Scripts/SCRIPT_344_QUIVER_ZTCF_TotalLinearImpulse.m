%Generate Club Quiver Plot
figure(344);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Linear Impulse on Club Quiver Plot
TotalLinearImpulse=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.LinearImpulseonClub(:,1),ZTCFQ.LinearImpulseonClub(:,2),ZTCFQ.LinearImpulseonClub(:,3));
TotalLinearImpulse.LineWidth=1;
TotalLinearImpulse.Color=[0 1 0];
TotalLinearImpulse.MaxHeadSize=0.1;
TotalLinearImpulse.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Total Linear Impulse');

%Add a Title
title('Total Linear Impulse on Club');
subtitle('ZTCF');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Total Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(344);

%Clear Figure from Workspace
clear TotalLinearImpulse;
clear MPSumofMomentsQuiver;
clear ZTCFTotalLinearImpulse;
clear ZTCFMPSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
