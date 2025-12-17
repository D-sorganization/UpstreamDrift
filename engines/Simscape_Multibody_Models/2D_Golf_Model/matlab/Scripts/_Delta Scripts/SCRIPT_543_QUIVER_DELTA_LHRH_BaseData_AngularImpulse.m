%Generate Club Quiver Plot
figure(543);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Angular Impulse Quiver Plot
LWAngularImpulseQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),DELTAQ.LWAngularImpulseonClub(:,1),DELTAQ.LWAngularImpulseonClub(:,2),DELTAQ.LWAngularImpulseonClub(:,3));
LWAngularImpulseQuiver.LineWidth=1;
LWAngularImpulseQuiver.Color=[0 1 0];
LWAngularImpulseQuiver.MaxHeadSize=0.1;
LWAngularImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Angular Impulse Quiver Plot
RWAngularImpulseQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),DELTAQ.RWAngularImpulseonClub(:,1),DELTAQ.RWAngularImpulseonClub(:,2),DELTAQ.RWAngularImpulseonClub(:,3));
RWAngularImpulseQuiver.LineWidth=1;
RWAngularImpulseQuiver.Color=[.8 .2 0];
RWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that LH and RH are scaled the same.
RWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/RWAngularImpulseQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','LW Angular Impulse','RW Angular Impulse');

%Add a Title
title('LW and RW Angular Impulse on Club');
subtitle('DELTA');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - LWRW Angular Impulse on Club');
pause(PauseTime);

%Close Figure
close(543);

%Clear Figure from Workspace
clear LWAngularImpulseQuiver;
clear RWAngularImpulseQuiver;
clear ZTCFLWAngularImpulseQuiver;
clear ZTCFRWAngularImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
