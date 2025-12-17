%Generate Club Quiver Plot
figure(703);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Equivalent Couple Quiver Plot
NetCoupleQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,2),BASEQ.EquivalentMidpointCoupleGlobal(:,3));
NetCoupleQuiver.LineWidth=1.5;
NetCoupleQuiver.Color=[0.07 0.62 1.0];
NetCoupleQuiver.MaxHeadSize=0.05;
NetCoupleQuiver.AutoScaleFactor=8;

%Generate ZTCF Equivalent Couple Quiver Plot
ZTCFNetCoupleQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,2),ZTCFQ.EquivalentMidpointCoupleGlobal(:,3));
ZTCFNetCoupleQuiver.LineWidth=1.5;
ZTCFNetCoupleQuiver.Color=[1.0 0.07 0.65];
ZTCFNetCoupleQuiver.MaxHeadSize=0.05;
%Correct scaling on Equivalent Couple so that ZTCF and BASE are scaled the same.
ZTCFNetCoupleQuiver.AutoScaleFactor=NetCoupleQuiver.ScaleFactor/ZTCFNetCoupleQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','BASE - Equivalent MP Couple','ZTCF - Equivalent MP Couple');

%Add a Title
title('Equivalent MP Couple');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Equivalent MP Couple');
pause(PauseTime);

%Close Figure
close(703);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear ZTCFNetForceQuiver;
clear ZTCFNetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
