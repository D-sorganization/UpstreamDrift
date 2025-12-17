%Generate Club Quiver Plot
figure(943);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Left Hand Angular Impulse Quiver Plot
LWAngularImpulseQuiver=quiver3(Data.LWx(:,1),Data.LWy(:,1),Data.LWz(:,1),Data.LWAngularImpulseonClub(:,1),Data.LWAngularImpulseonClub(:,2),Data.LWAngularImpulseonClub(:,3));
LWAngularImpulseQuiver.LineWidth=1;
LWAngularImpulseQuiver.Color=[0 1 0];
LWAngularImpulseQuiver.MaxHeadSize=0.1;
LWAngularImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Angular Impulse Quiver Plot
RWAngularImpulseQuiver=quiver3(Data.RWx(:,1),Data.RWy(:,1),Data.RWz(:,1),Data.RWAngularImpulseonClub(:,1),Data.RWAngularImpulseonClub(:,2),Data.RWAngularImpulseonClub(:,3));
RWAngularImpulseQuiver.LineWidth=1;
RWAngularImpulseQuiver.Color=[.8 .2 0];
RWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that LH and RH are scaled the same.
RWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/RWAngularImpulseQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','LW Angular Impulse','RW Angular Impulse');

%Add a Title
title('LW and RW Angular Impulse on Club');
subtitle('Data');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Data Quiver Plots/Data_Quiver Plot - LWRW Angular Impulse on Club');
pause(PauseTime);

%Close Figure
close(943);

%Clear Figure from Workspace
clear LWAngularImpulseQuiver;
clear RWAngularImpulseQuiver;
clear ZTCFLWAngularImpulseQuiver;
clear ZTCFRWAngularImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
