%Generate Club Quiver Plot
figure(116);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LH MOF Quiver Plot
LHMOFQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LHMOFonClubGlobal(:,1),BASEQ.LHMOFonClubGlobal(:,2),BASEQ.LHMOFonClubGlobal(:,3));
LHMOFQuiver.LineWidth=1;
LHMOFQuiver.Color=[0 0.5 0];
LHMOFQuiver.MaxHeadSize=0.1;
LHMOFQuiver.AutoScaleFactor=4;

%Generate LW Torque Quiver Plot
LHTorqueQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWonClubTGlobal(:,1),BASEQ.LWonClubTGlobal(:,2),BASEQ.LWonClubTGlobal(:,3));
LHTorqueQuiver.LineWidth=1;
LHTorqueQuiver.Color=[0 0 1];
LHTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
LHTorqueQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/LHTorqueQuiver.ScaleFactor;

%Generate RH MOF Quiver Plot
RHMOFQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RHMOFonClubGlobal(:,1),BASEQ.RHMOFonClubGlobal(:,2),BASEQ.RHMOFonClubGlobal(:,3));
RHMOFQuiver.LineWidth=1;
RHMOFQuiver.Color=[0.5 0 0];
RHMOFQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
RHMOFQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/RHMOFQuiver.ScaleFactor;

%Generate RW Total Torque Quiver Plot
RHTorqueQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RWonClubTGlobal(:,1),BASEQ.RWonClubTGlobal(:,2),BASEQ.RWonClubTGlobal(:,3));
RHTorqueQuiver.LineWidth=1;
RHTorqueQuiver.Color=[1 0 0];
RHTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
RHTorqueQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/RHTorqueQuiver.ScaleFactor;

%Generate Total MOF Quiver Plot
TotalMOFQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.MPMOFonClubGlobal(:,1),BASEQ.MPMOFonClubGlobal(:,2),BASEQ.MPMOFonClubGlobal(:,3));
TotalMOFQuiver.LineWidth=1;
TotalMOFQuiver.Color=[0 0 0.5];
TotalMOFQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
TotalMOFQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/TotalMOFQuiver.ScaleFactor;

%Generate Total Torque Quiver Plot
NetTorqueQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalWristTorqueGlobal(:,1),BASEQ.TotalWristTorqueGlobal(:,2),BASEQ.TotalWristTorqueGlobal(:,3));
NetTorqueQuiver.LineWidth=1;
NetTorqueQuiver.Color=[0 1 0];
NetTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
NetTorqueQuiver.AutoScaleFactor=LHMOFQuiver.ScaleFactor/NetTorqueQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH MOF','LH Torque','RH MOF','RH Torque','Total MOF','Net Torque');

%Add a Title
title('All Torques and Moments on Club');
subtitle('BASE');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('BaseData Quiver Plots//BASE_Quiver Plot - Torques and Moments');
pause(PauseTime);

%Close Figure
close(116);

%Clear Figure from Workspace
clear LHTorqueQuiver;
clear RHTorqueQuiver;
clear NetTorqueQuiver;
clear LHMOFQuiver;
clear RHMOFQuiver;
clear TotalMOFQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;