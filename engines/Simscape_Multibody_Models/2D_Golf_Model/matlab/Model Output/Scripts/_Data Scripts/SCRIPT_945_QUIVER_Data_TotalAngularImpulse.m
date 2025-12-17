%Generate Club Quiver Plot
figure(945);
run SCRIPT_QuiverClubandShaftData.m;

%Create Total Angular Impulse Array by Adding LH and RH Angular Impulse
TotalAngularImpulse=Data.LWAngularImpulseonClub+Data.RWAngularImpulseonClub;

%Generate Total Hand Angular Impulse Quiver Plot
AngularImpulseQuiver=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),TotalAngularImpulse(:,1),TotalAngularImpulse(:,2),TotalAngularImpulse(:,3));
AngularImpulseQuiver.LineWidth=1;
AngularImpulseQuiver.Color=[0 1 0];
AngularImpulseQuiver.MaxHeadSize=0.1;
AngularImpulseQuiver.AutoScaleFactor=3;

%Add Legend to Plot
legend('','','','','Total Angular Impulse');

%Add a Title
title('Total Angular Impulse on Club');
subtitle('Data');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Data Quiver Plots/Data_Quiver Plot - Total Angular Impulse on Club');
pause(PauseTime);

%Close Figure
close(945);

%Clear Figure from Workspace
clear AngularImpulseQuiver;
clear RWAngularImpulseQuiver;
clear ZTCFAngularImpulseQuiver;
clear ZTCFRWAngularImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
clear TotalAngularImpulse;
