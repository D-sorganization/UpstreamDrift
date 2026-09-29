%Generate Club Quiver Plot
figure(725);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Create Total Angular Impulse Array by Adding LH and RH Angular Impulse
TotalAngularImpulse=BASEQ.LWAngularImpulseonClub+BASEQ.RWAngularImpulseonClub;
ZTCFTotalAngularImpulse=ZTCFQ.LWAngularImpulseonClub+ZTCFQ.RWAngularImpulseonClub;

%Generate Total Hand Angular Impulse Quiver Plot
AngularImpulseQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),TotalAngularImpulse(:,1),TotalAngularImpulse(:,2),TotalAngularImpulse(:,3));
AngularImpulseQuiver.LineWidth=1;
AngularImpulseQuiver.Color=[0 1 0];
AngularImpulseQuiver.MaxHeadSize=0.1;
AngularImpulseQuiver.AutoScaleFactor=3;

%Generate ZTCF Total Hand Angular Impulse Quiver Plot
ZTCFAngularImpulseQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFTotalAngularImpulse(:,1),ZTCFTotalAngularImpulse(:,2),ZTCFTotalAngularImpulse(:,3));
ZTCFAngularImpulseQuiver.LineWidth=1;
ZTCFAngularImpulseQuiver.Color=[0 0.3 0];
ZTCFAngularImpulseQuiver.MaxHeadSize=0.1;
%Correct scaling on Angular Impulses so that ZTCF and BASE are scaled the same.
ZTCFAngularImpulseQuiver.AutoScaleFactor=AngularImpulseQuiver.ScaleFactor/ZTCFAngularImpulseQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','BASE - Total Impulse','ZTCF - Total Impulse');

%Add a Title
title('Total Impulse on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Total Impulse on Club');
pause(PauseTime);

%Close Figure
close(725);

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
clear ZTCFTotalAngularImpulse;