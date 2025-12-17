%Generate Club Quiver Plot
figure(544);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Linear Impulse on Club Quiver Plot
TotalLinearImpulse=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),DELTAQ.LinearImpulseonClub(:,1),DELTAQ.LinearImpulseonClub(:,2),DELTAQ.LinearImpulseonClub(:,3));
TotalLinearImpulse.LineWidth=1;
TotalLinearImpulse.Color=[0 1 0];
TotalLinearImpulse.MaxHeadSize=0.1;
TotalLinearImpulse.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Total Linear Impulse');

%Add a Title
title('Total Linear Impulse on Club');
subtitle('DELTA');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Total Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(544);

%Clear Figure from Workspace
clear TotalLinearImpulse;
clear MPSumofMomentsQuiver;
clear ZTCFTotalLinearImpulse;
clear ZTCFMPSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
