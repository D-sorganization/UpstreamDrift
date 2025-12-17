%Generate Club Quiver Plot
figure(4);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate LW Total Torque Quiver Plot
LHTorqueQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWonClubTGlobal(:,1),BASEQ.LWonClubTGlobal(:,2),BASEQ.LWonClubTGlobal(:,3));
LHTorqueQuiver.LineWidth=1;
LHTorqueQuiver.Color=[0 0 1];
LHTorqueQuiver.AutoScaleFactor=2;
LHTorqueQuiver.MaxHeadSize=0.1;

%Generate RW Total Torque Quiver Plot
RHTorqueQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.RWonClubTGlobal(:,1),BASEQ.RWonClubTGlobal(:,2),BASEQ.RWonClubTGlobal(:,3));
RHTorqueQuiver.LineWidth=1;
RHTorqueQuiver.Color=[1 0 0];
RHTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
RHTorqueQuiver.AutoScaleFactor=LHTorqueQuiver.ScaleFactor/RHTorqueQuiver.ScaleFactor;

%Generate Total Torque Quiver Plot
NetTorqueQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalWristTorqueGlobal(:,1),BASEQ.TotalWristTorqueGlobal(:,2),BASEQ.TotalWristTorqueGlobal(:,3));
NetTorqueQuiver.LineWidth=1;
NetTorqueQuiver.Color=[0 1 0];
NetTorqueQuiver.MaxHeadSize=0.1;
%Correct scaling so that all are scaled the same.
NetTorqueQuiver.AutoScaleFactor=LHTorqueQuiver.ScaleFactor/NetTorqueQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','LH Torque','RH Torque','Net Torque');

%Add a Title
title('Total Torque');
subtitle('COMPARISON');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Comparison Quiver Plots//COMPARISON_Quiver Plot - Hand Torques');

%Close Figure
close(4);

%Clear Figure from Workspace
clear LHTorqueQuiver;
clear RHTorqueQuiver;
clear NetTorqueQuiver;
clear Grip;
clear Shaft;