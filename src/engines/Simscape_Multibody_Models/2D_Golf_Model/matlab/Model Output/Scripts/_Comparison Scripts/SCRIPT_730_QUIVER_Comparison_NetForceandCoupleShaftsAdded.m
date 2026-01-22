%Generate Club Quiver Plot
figure(730);
run SCRIPT_QuiverClubandShaftBaseData.m;
run SCRIPT_ShaftandGripatMaxCHSandAlphaReversal.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalHandForceGlobal(:,1),BASEQ.TotalHandForceGlobal(:,2),BASEQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;

%Generate Equivalent Couple Quiver Plot
NetCoupleQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,2),BASEQ.EquivalentMidpointCoupleGlobal(:,3));
NetCoupleQuiver.LineWidth=1;
NetCoupleQuiver.Color=[.8 .2 0];
NetCoupleQuiver.MaxHeadSize=0.1;
NetCoupleQuiver.AutoScaleFactor=3;

%Generate ZTCF Total Force Quiver Plot
ZTCFNetForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.TotalHandForceGlobal(:,1),ZTCFQ.TotalHandForceGlobal(:,2),ZTCFQ.TotalHandForceGlobal(:,3));
ZTCFNetForceQuiver.LineWidth=1;
ZTCFNetForceQuiver.Color=[0 0.3 0];
ZTCFNetForceQuiver.MaxHeadSize=0.1;
%Correct scaling on Forces so that ZTCF and BASE are scaled the same.
ZTCFNetForceQuiver.AutoScaleFactor=NetForceQuiver.ScaleFactor/ZTCFNetForceQuiver.ScaleFactor;

%Generate ZTCF Equivalent Couple Quiver Plot
ZTCFNetCoupleQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,2),ZTCFQ.EquivalentMidpointCoupleGlobal(:,3));
ZTCFNetCoupleQuiver.LineWidth=1;
ZTCFNetCoupleQuiver.Color=[.5 .5 0];
ZTCFNetCoupleQuiver.MaxHeadSize=0.1;
%Correct scaling on Equivalent Couple so that ZTCF and BASE are scaled the same.
ZTCFNetCoupleQuiver.AutoScaleFactor=NetCoupleQuiver.ScaleFactor/ZTCFNetCoupleQuiver.ScaleFactor;


%Add Legend to Plot
legend('','','','','','','','','BASE - Net Force','BASE - Equivalent MP Couple','ZTCF - Net Force','ZTCF - Equivalent MP Couple');

%Add a Title
title('Net Force and Equivalent MP Couple');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Net Force and Equivalent MP Couple Shafts Added');
pause(PauseTime);

%Close Figure
close(730);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear ZTCFNetForceQuiver;
clear ZTCFNetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
clear GripAlpha;
clear GripMaxCHS;
clear ShaftAlpha;
clear ShaftMaxCHS;

