%Generate Club Quiver Plot
figure(144);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Linear Impulse on Club Quiver Plot
TotalLinearImpulse=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.LinearImpulseonClub(:,1),BASEQ.LinearImpulseonClub(:,2),BASEQ.LinearImpulseonClub(:,3));
TotalLinearImpulse.LineWidth=1;
TotalLinearImpulse.Color=[0 1 0];
TotalLinearImpulse.MaxHeadSize=0.1;
TotalLinearImpulse.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Total Linear Impulse');

%Add a Title
title('Total Linear Impulse on Club');
subtitle('BASE');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Total Linear Impulse on Club');
pause(PauseTime);

%Close Figure
close(144);

%Clear Figure from Workspace
clear TotalLinearImpulse;
clear MPSumofMomentsQuiver;
clear ZTCFTotalLinearImpulse;
clear ZTCFMPSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
