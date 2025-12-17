%Generate Club Quiver Plot
figure(721);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Moment of Force Quiver Plot
TotalMomentofForceQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.MPMOFonClubGlobal(:,1),BASEQ.MPMOFonClubGlobal(:,2),BASEQ.MPMOFonClubGlobal(:,3));
TotalMomentofForceQuiver.LineWidth=1;
TotalMomentofForceQuiver.Color=[0 1 0];
TotalMomentofForceQuiver.MaxHeadSize=0.1;
TotalMomentofForceQuiver.AutoScaleFactor=3;

%Generate Equivalent Couple Quiver Plot
EquivalentCoupleQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,2),BASEQ.EquivalentMidpointCoupleGlobal(:,3));
EquivalentCoupleQuiver.LineWidth=1;
EquivalentCoupleQuiver.Color=[.8 .2 0];
EquivalentCoupleQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that ZTCF and BASE are scaled the same.
EquivalentCoupleQuiver.AutoScaleFactor=TotalMomentofForceQuiver.ScaleFactor/EquivalentCoupleQuiver.ScaleFactor;

%Generate ZTCF Total Moment of Force Quiver Plot
ZTCFTotalMomentofForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.MPMOFonClubGlobal(:,1),ZTCFQ.MPMOFonClubGlobal(:,2),ZTCFQ.MPMOFonClubGlobal(:,3));
ZTCFTotalMomentofForceQuiver.LineWidth=1;
ZTCFTotalMomentofForceQuiver.Color=[0 0.3 0];
ZTCFTotalMomentofForceQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Forces so that ZTCF and BASE are scaled the same.
ZTCFTotalMomentofForceQuiver.AutoScaleFactor=TotalMomentofForceQuiver.ScaleFactor/ZTCFTotalMomentofForceQuiver.ScaleFactor;

%Generate ZTCF Equivalent Couple Quiver Plot
ZTCFEquivalentCoupleQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,1),ZTCFQ.EquivalentMidpointCoupleGlobal(:,2),ZTCFQ.EquivalentMidpointCoupleGlobal(:,3));
ZTCFEquivalentCoupleQuiver.LineWidth=1;
ZTCFEquivalentCoupleQuiver.Color=[.5 .5 0];
ZTCFEquivalentCoupleQuiver.MaxHeadSize=0.1;
%Correct scaling on Moment of Force so that ZTCF and BASE are scaled the same.
ZTCFEquivalentCoupleQuiver.AutoScaleFactor=TotalMomentofForceQuiver.ScaleFactor/ZTCFEquivalentCoupleQuiver.ScaleFactor;



%Add Legend to Plot
legend('','','','','BASE - Moment of Force','BASE - Equivalent Couple','ZTCF - Moment of Force','ZTCF - Equivalent Couple');

%Add a Title
title('Total MOF and Equivalent Couple on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Total MOF and Equivalent Couple');
pause(PauseTime);

%Close Figure
close(721);

%Clear Figure from Workspace
clear TotalMomentofForceQuiver;
clear EquivalentCoupleQuiver;
clear ZTCFTotalMomentofForceQuiver;
clear ZTCFEquivalentCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
