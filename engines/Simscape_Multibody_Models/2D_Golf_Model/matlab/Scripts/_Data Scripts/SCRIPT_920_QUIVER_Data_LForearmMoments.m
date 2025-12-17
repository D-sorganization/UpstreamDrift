%Generate Club Quiver Plot
figure(920);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Club on LW Total Force Quiver Plot
LHForceQuiver=quiver3(Data.LWx(:,1),Data.LWy(:,1),Data.LWz(:,1),Data.ClubonLWFGlobal(:,1),Data.ClubonLWFGlobal(:,2),Data.ClubonLWFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate LE Total Force on L Forearm Quiver Plot
LEForceQuiver=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LArmonLForearmFGlobal(:,1),Data.LArmonLForearmFGlobal(:,2),Data.LArmonLForearmFGlobal(:,3));
LEForceQuiver.LineWidth=1;
LEForceQuiver.Color=[1 0 0];
LEForceQuiver.MaxHeadSize=0.1;
LEForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/LEForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Forearm
LEMOFLForearmQuiver=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LElbowonLForearmMOFGlobal(:,1),Data.LElbowonLForearmMOFGlobal(:,2),Data.LElbowonLForearmMOFGlobal(:,3));
LEMOFLForearmQuiver.LineWidth=1;
LEMOFLForearmQuiver.Color=[0 0.75 0];
LEMOFLForearmQuiver.MaxHeadSize=0.1;
LEMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Left Wrist MOF on Left Forearm
LWristMOFLForearm=quiver3(Data.LWx(:,1),Data.LWy(:,1),Data.LWz(:,1),Data.LWristonLForearmMOFGlobal(:,1),Data.LWristonLForearmMOFGlobal(:,2),Data.LWristonLForearmMOFGlobal(:,3));
LWristMOFLForearm.LineWidth=1;
LWristMOFLForearm.Color=[0 0.5 0];
LWristMOFLForearm.MaxHeadSize=0.1;
LWristMOFLForearm.AutoScaleFactor=LEMOFLForearmQuiver.ScaleFactor/LWristMOFLForearm.ScaleFactor;

%Generate LeftForearm Quivers
LeftForearm=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LeftForearmdx(:,1),Data.LeftForearmdy(:,1),Data.LeftForearmdz(:,1),0);
LeftForearm.ShowArrowHead='off';		
LeftForearm.LineWidth=1;			   
LeftForearm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','LH Force','LE Force','LE MOF','LH MOF','');

%Add a Title
title('Moments of Force Acting on Left Forearm');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Left Forearm Moments');
pause(PauseTime);

%Close Figure
close(920);

%Clear Figure from Workspace
clear LHForceQuiver;
clear LEForceQuiver;
clear LEMOFLForearmQuiver;
clear LWristMOFLForearm;
clear LeftForearm;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;