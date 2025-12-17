%Generate Club Quiver Plot
figure(343);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Angular Impulse Quiver Plot
LWAngularImpulseQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LWAngularImpulseonClub(:,1),ZTCFQ.LWAngularImpulseonClub(:,2),ZTCFQ.LWAngularImpulseonClub(:,3));
LWAngularImpulseQuiver.LineWidth=1;
LWAngularImpulseQuiver.Color=[0 1 0];
LWAngularImpulseQuiver.MaxHeadSize=0.1;
LWAngularImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Angular Impulse Quiver Plot
RWAngularImpulseQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RWAngularImpulseonClub(:,1),ZTCFQ.RWAngularImpulseonClub(:,2),ZTCFQ.RWAngularImpulseonClub(:,3));
RWAngularImpulseQuiver.LineWidth=1;
RWAngularImpulseQuiver.Color=[.8 .2 0];
RWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that LH and RH are scaled the same.
RWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/RWAngularImpulseQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','LW Angular Impulse','RW Angular Impulse');

%Add a Title
title('LW and RW Angular Impulse on Club');
subtitle('ZTCF');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - LWRW Angular Impulse on Club');
pause(PauseTime);

%Close Figure
close(343);

%Clear Figure from Workspace
clear LWAngularImpulseQuiver;
clear RWAngularImpulseQuiver;
clear ZTCFLWAngularImpulseQuiver;
clear ZTCFRWAngularImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
