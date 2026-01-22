%Generate Club Quiver Plot
figure(149);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LH MOF Quiver Plot
LHMOFQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LHMOFonClubGlobal(:,1),BASEQ.LHMOFonClubGlobal(:,2),BASEQ.LHMOFonClubGlobal(:,3));
LHMOFQuiver.LineWidth=1;
LHMOFQuiver.Color=[0 0.5 0];
LHMOFQuiver.MaxHeadSize=0.1;
LHMOFQuiver.AutoScaleFactor=10;

%Generate RH MOF Quiver Plot
RHMOFQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RHMOFonClubGlobal(:,1),BASEQ.RHMOFonClubGlobal(:,2),BASEQ.RHMOFonClubGlobal(:,3));
RHMOFQuiver.LineWidth=1;
RHMOFQuiver.Color=[0.5 0 0];
RHMOFQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
RHMOFQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/RHMOFQuiver.ScaleFactor;

%Generate Total MOF Quiver Plot
% TotalMOFQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.MPMOFonClubGlobal(:,1),BASEQ.MPMOFonClubGlobal(:,2),BASEQ.MPMOFonClubGlobal(:,3));
% TotalMOFQuiver.LineWidth=1;
% TotalMOFQuiver.Color=[0 0 0.5];
% TotalMOFQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
% TotalMOFQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/TotalMOFQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH MOF','RH MOF');

%Add a Title
title('LHRH Moments of Force on Club');
subtitle('BASE');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('BaseData Quiver Plots/BASE_Quiver Plot - LHRH Moments of Force');
pause(PauseTime);

%Close Figure
close(149);

%Clear Figure from Workspace
clear LHMOFQuiver;
clear RHMOFQuiver;
clear TotalMOFQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;