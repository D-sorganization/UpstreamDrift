%Generate Club Quiver Plot
figure(320);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Club on LW Total Force Quiver Plot
LHForceQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.ClubonLWFGlobal(:,1),ZTCFQ.ClubonLWFGlobal(:,2),ZTCFQ.ClubonLWFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate LE Total Force on L Forearm Quiver Plot
LEForceQuiver=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),ZTCFQ.LArmonLForearmFGlobal(:,1),ZTCFQ.LArmonLForearmFGlobal(:,2),ZTCFQ.LArmonLForearmFGlobal(:,3));
LEForceQuiver.LineWidth=1;
LEForceQuiver.Color=[1 0 0];
LEForceQuiver.MaxHeadSize=0.1;
LEForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/LEForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Forearm
LEMOFLForearmQuiver=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),ZTCFQ.LElbowonLForearmMOFGlobal(:,1),ZTCFQ.LElbowonLForearmMOFGlobal(:,2),ZTCFQ.LElbowonLForearmMOFGlobal(:,3));
LEMOFLForearmQuiver.LineWidth=1;
LEMOFLForearmQuiver.Color=[0 0.75 0];
LEMOFLForearmQuiver.MaxHeadSize=0.1;
LEMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Left Wrist MOF on Left Forearm
LWristMOFLForearm=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.LWristonLForearmMOFGlobal(:,1),ZTCFQ.LWristonLForearmMOFGlobal(:,2),ZTCFQ.LWristonLForearmMOFGlobal(:,3));
LWristMOFLForearm.LineWidth=1;
LWristMOFLForearm.Color=[0 0.5 0];
LWristMOFLForearm.MaxHeadSize=0.1;
LWristMOFLForearm.AutoScaleFactor=LEMOFLForearmQuiver.ScaleFactor/LWristMOFLForearm.ScaleFactor;

%Generate LeftForearm Quivers
LeftForearm=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),ZTCFQ.LeftForearmdx(:,1),ZTCFQ.LeftForearmdy(:,1),ZTCFQ.LeftForearmdz(:,1),0);
LeftForearm.ShowArrowHead='off';		
LeftForearm.LineWidth=1;			   
LeftForearm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','LH Force','LE Force','LE MOF','LH MOF','');

%Add a Title
title('Moments of Force Acting on Left Forearm');
subtitle('ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Left Forearm Moments');
pause(PauseTime);

%Close Figure
close(320);

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