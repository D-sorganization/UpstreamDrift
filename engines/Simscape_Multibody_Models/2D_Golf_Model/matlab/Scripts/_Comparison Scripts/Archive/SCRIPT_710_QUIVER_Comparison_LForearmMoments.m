%Generate Club Quiver Plot
figure(9);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Club on LW Total Force Quiver Plot
LHForceQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.ClubonLWFGlobal(:,1),BASEQ.ClubonLWFGlobal(:,2),BASEQ.ClubonLWFGlobal(:,3));
LHForceQuiver.LineWidth=1;
LHForceQuiver.Color=[0 0 1];
LHForceQuiver.AutoScaleFactor=2;
LHForceQuiver.MaxHeadSize=0.1;

%Generate LE Total Force on L Forearm Quiver Plot
LEForceQuiver=quiver3(BASEQ.LEx(:,1),BASEQ.LEy(:,1),BASEQ.LEz(:,1),BASEQ.LArmonLForearmFGlobal(:,1),BASEQ.LArmonLForearmFGlobal(:,2),BASEQ.LArmonLForearmFGlobal(:,3));
LEForceQuiver.LineWidth=1;
LEForceQuiver.Color=[1 0 0];
LEForceQuiver.MaxHeadSize=0.1;
LEForceQuiver.AutoScaleFactor=LHForceQuiver.ScaleFactor/LEForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Forearm
LEMOFLForearmQuiver=quiver3(BASEQ.LEx(:,1),BASEQ.LEy(:,1),BASEQ.LEz(:,1),BASEQ.LElbowonLForearmMOFGlobal(:,1),BASEQ.LElbowonLForearmMOFGlobal(:,2),BASEQ.LElbowonLForearmMOFGlobal(:,3));
LEMOFLForearmQuiver.LineWidth=1;
LEMOFLForearmQuiver.Color=[0 0.75 0];
LEMOFLForearmQuiver.MaxHeadSize=0.1;
LEMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Left Wrist MOF on Left Forearm
LWristMOFLForearm=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.LWristonLForearmMOFGlobal(:,1),BASEQ.LWristonLForearmMOFGlobal(:,2),BASEQ.LWristonLForearmMOFGlobal(:,3));
LWristMOFLForearm.LineWidth=1;
LWristMOFLForearm.Color=[0 0.5 0];
LWristMOFLForearm.MaxHeadSize=0.1;
LWristMOFLForearm.AutoScaleFactor=LEMOFLForearmQuiver.ScaleFactor/LWristMOFLForearm.ScaleFactor;

%Generate LeftForearm Quivers
LeftForearm=quiver3(BASEQ.LEx(:,1),BASEQ.LEy(:,1),BASEQ.LEz(:,1),BASEQ.LeftForearmdx(:,1),BASEQ.LeftForearmdy(:,1),BASEQ.LeftForearmdz(:,1),0);
LeftForearm.ShowArrowHead='off';		
LeftForearm.LineWidth=1;			   
LeftForearm.Color=[0 0 0];

%Add Legend to Plot
legend('','','LH Force','LE Force','LE MOF','LH MOF','');

%Add a Title
title('Moments of Force Acting on Left Forearm');
subtitle('COMPARISON');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Left Forearm Moments');

%Close Figure
close(9);

%Clear Figure from Workspace
clear LHForceQuiver;
clear LEForceQuiver;
clear LEMOFLForearmQuiver;
clear LWristMOFLForearm;
clear LeftForearm;
clear Grip;
clear Shaft;