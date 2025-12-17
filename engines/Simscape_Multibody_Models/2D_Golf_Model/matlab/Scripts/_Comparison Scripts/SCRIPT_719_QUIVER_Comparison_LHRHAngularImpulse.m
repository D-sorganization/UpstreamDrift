%Generate Club Quiver Plot
figure(719);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Angular Impulse Quiver Plot
LWAngularImpulseQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWAngularImpulseonClub(:,1),BASEQ.LWAngularImpulseonClub(:,2),BASEQ.LWAngularImpulseonClub(:,3));
LWAngularImpulseQuiver.LineWidth=1;
LWAngularImpulseQuiver.Color=[0 1 0];
LWAngularImpulseQuiver.MaxHeadSize=0.1;
LWAngularImpulseQuiver.AutoScaleFactor=3;

%Generate Right Hand Angular Impulse Quiver Plot
RWAngularImpulseQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RWAngularImpulseonClub(:,1),BASEQ.RWAngularImpulseonClub(:,2),BASEQ.RWAngularImpulseonClub(:,3));
RWAngularImpulseQuiver.LineWidth=1;
RWAngularImpulseQuiver.Color=[.8 .2 0];
RWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that ZTCF and BASE are scaled the same.
RWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/RWAngularImpulseQuiver.ScaleFactor;


%Generate ZTCF Left Hand Angular Impulse Quiver Plot
ZTCFLWAngularImpulseQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LWAngularImpulseonClub(:,1),ZTCFQ.LWAngularImpulseonClub(:,2),ZTCFQ.LWAngularImpulseonClub(:,3));
ZTCFLWAngularImpulseQuiver.LineWidth=1;
ZTCFLWAngularImpulseQuiver.Color=[0 0.3 0];
ZTCFLWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that ZTCF and BASE are scaled the same.
ZTCFLWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/ZTCFLWAngularImpulseQuiver.ScaleFactor;

%Generate ZTCF Right Hand Angular Impulse Quiver Plot
ZTCFRWAngularImpulseQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RWAngularImpulseonClub(:,1),ZTCFQ.RWAngularImpulseonClub(:,2),ZTCFQ.RWAngularImpulseonClub(:,3));
ZTCFRWAngularImpulseQuiver.LineWidth=1;
ZTCFRWAngularImpulseQuiver.Color=[.5 .5 0];
ZTCFRWAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulse so that ZTCF and BASE are scaled the same.
ZTCFRWAngularImpulseQuiver.AutoScaleFactor=LWAngularImpulseQuiver.ScaleFactor/ZTCFRWAngularImpulseQuiver.ScaleFactor;



%Add Legend to Plot
legend('','','','','BASE - LW Angular Impulse','BASE - RW Angular Impulse','ZTCF - LW Angular Impulse','ZTCF - RW Angular Impulse');

%Add a Title
title('LW and RW Angular Impulse on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - LWRW Angular Impulse on Club');
pause(PauseTime);

%Close Figure
close(719);

%Clear Figure from Workspace
clear LWAngularImpulseQuiver;
clear RWAngularImpulseQuiver;
clear ZTCFLWAngularImpulseQuiver;
clear ZTCFRWAngularImpulseQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
