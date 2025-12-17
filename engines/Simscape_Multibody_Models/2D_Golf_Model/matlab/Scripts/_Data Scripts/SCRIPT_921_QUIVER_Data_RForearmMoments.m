%Generate Club Quiver Plot
figure(921);
run SCRIPT_QuiverClubandShaftData.m;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(Data.RWx(:,1),Data.RWy(:,1),Data.RWz(:,1),Data.ClubonRWFGlobal(:,1),Data.ClubonRWFGlobal(:,2),Data.ClubonRWFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[0 0 1];
RHForceQuiver.AutoScaleFactor=2;
RHForceQuiver.MaxHeadSize=0.1;

%Generate RE Total Force Quiver Plot
REForceQuiver=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RArmonRForearmFGlobal(:,1),Data.RArmonRForearmFGlobal(:,2),Data.RArmonRForearmFGlobal(:,3));
REForceQuiver.LineWidth=1;
REForceQuiver.Color=[1 0 0];
REForceQuiver.MaxHeadSize=0.1;
REForceQuiver.AutoScaleFactor=RHForceQuiver.ScaleFactor/REForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Forearm
REMOFLForearmQuiver=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RElbowonRForearmMOFGlobal(:,1),Data.RElbowonRForearmMOFGlobal(:,2),Data.RElbowonRForearmMOFGlobal(:,3));
REMOFLForearmQuiver.LineWidth=1;
REMOFLForearmQuiver.Color=[0 0.75 0];
REMOFLForearmQuiver.MaxHeadSize=0.1;
REMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Right Wrist MOF on Right Forearm
RWristMOFLForearm=quiver3(Data.RWx(:,1),Data.RWy(:,1),Data.RWz(:,1),Data.RWristonRForearmMOFGlobal(:,1),Data.RWristonRForearmMOFGlobal(:,2),Data.RWristonRForearmMOFGlobal(:,3));
RWristMOFLForearm.LineWidth=1;
RWristMOFLForearm.Color=[0 0.5 0];
RWristMOFLForearm.MaxHeadSize=0.1;
RWristMOFLForearm.AutoScaleFactor=REMOFLForearmQuiver.ScaleFactor/RWristMOFLForearm.ScaleFactor;

%Generate Right Forearm Quivers
RightForearm=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RightForearmdx(:,1),Data.RightForearmdy(:,1),Data.RightForearmdz(:,1),0);
RightForearm.ShowArrowHead='off';
RightForearm.LineWidth=1;			   
RightForearm.Color=[0 0 0];	

%Add Legend to Plot
legend('','','','','RH Force','RE Force','RE MOF','RH MOF');

%Add a Title
title('Moments of Force Acting on Right Forearm');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Right Forearm Moments');
pause(PauseTime);

%Close Figure
close(921);

%Clear Figure from Workspace
clear RHForceQuiver;
clear REForceQuiver;
clear REMOFLForearmQuiver;
clear RWristMOFLForearm;
clear RightForearm;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;