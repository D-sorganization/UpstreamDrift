%Generate Club Quiver Plot
figure(944);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Total Linear Impulse on Club Quiver Plot
TotalLinearImpulse=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),Data.LinearImpulseonClub(:,1),Data.LinearImpulseonClub(:,2),Data.LinearImpulseonClub(:,3));
TotalLinearImpulse.LineWidth=1;
TotalLinearImpulse.Color=[0 1 0];
TotalLinearImpulse.MaxHeadSize=0.1;
TotalLinearImpulse.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Total Linear Impulse');

%Add a Title
title('Total Linear Impulse on Club');
subtitle('Data');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Data Quiver Plots/Data_Quiver Plot - Total Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(944);

%Clear Figure from Workspace
clear TotalLinearImpulse;
clear MPSumofMomentsQuiver;
clear ZTCFTotalLinearImpulse;
clear ZTCFMPSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
