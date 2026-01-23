%Generate Club Quiver Plot
figure(515);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LW Total Torque Quiver Plot
LHTorqueQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),DELTAQ.LWonClubTGlobal(:,1),DELTAQ.LWonClubTGlobal(:,2),DELTAQ.LWonClubTGlobal(:,3));
LHTorqueQuiver.LineWidth=1;
LHTorqueQuiver.Color=[0 0 1];
LHTorqueQuiver.AutoScaleFactor=2;
LHTorqueQuiver.MaxHeadSize=0.1;

%Generate RW Total Torque Quiver Plot
RHTorqueQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),DELTAQ.RWonClubTGlobal(:,1),DELTAQ.RWonClubTGlobal(:,2),DELTAQ.RWonClubTGlobal(:,3));
RHTorqueQuiver.LineWidth=1;
RHTorqueQuiver.Color=[1 0 0];
RHTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
RHTorqueQuiver.AutoScaleFactor=LHTorqueQuiver.ScaleFactor/RHTorqueQuiver.ScaleFactor;

%Generate Total Torque Quiver Plot
NetTorqueQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),DELTAQ.TotalWristTorqueGlobal(:,1),DELTAQ.TotalWristTorqueGlobal(:,2),DELTAQ.TotalWristTorqueGlobal(:,3));
NetTorqueQuiver.LineWidth=1;
NetTorqueQuiver.Color=[0 1 0];
NetTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
NetTorqueQuiver.AutoScaleFactor=LHTorqueQuiver.ScaleFactor/NetTorqueQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','LH Torque','RH Torque','Net Torque');

%Add a Title
title('Total Torque');
subtitle('DELTA');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Hand Torques');
pause(PauseTime);

%Close Figure
close(515);

%Clear Figure from Workspace
clear LHTorqueQuiver;
clear RHTorqueQuiver;
clear NetTorqueQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;