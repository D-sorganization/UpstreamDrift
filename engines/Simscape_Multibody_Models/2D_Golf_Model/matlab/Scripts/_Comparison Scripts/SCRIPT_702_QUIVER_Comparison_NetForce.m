%Generate Club Quiver Plot
figure(702);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalHandForceGlobal(:,1),BASEQ.TotalHandForceGlobal(:,2),BASEQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=2;
NetForceQuiver.Color=[1 0 1];
NetForceQuiver.MaxHeadSize=0.075;
NetForceQuiver.AutoScaleFactor=8;


%Generate ZTCF Total Force Quiver Plot
ZTCFNetForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.TotalHandForceGlobal(:,1),ZTCFQ.TotalHandForceGlobal(:,2),ZTCFQ.TotalHandForceGlobal(:,3));
ZTCFNetForceQuiver.LineWidth=2;
ZTCFNetForceQuiver.Color=[0.07 0.62 1.0];
ZTCFNetForceQuiver.MaxHeadSize=0.075;
%Correct scaling on Forces so that ZTCF and BASE are scaled the same.
ZTCFNetForceQuiver.AutoScaleFactor=NetForceQuiver.ScaleFactor/ZTCFNetForceQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','BASE - Net Force','ZTCF - Net Force');

%Add a Title
title('Net Force');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Net Force');
pause(PauseTime);

%Close Figure
close(702);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear ZTCFNetForceQuiver;
clear ZTCFNetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
