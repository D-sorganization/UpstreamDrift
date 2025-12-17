%Generate Club Quiver Plot
figure(321);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.ClubonRWFGlobal(:,1),ZTCFQ.ClubonRWFGlobal(:,2),ZTCFQ.ClubonRWFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[0 0 1];
RHForceQuiver.AutoScaleFactor=2;
RHForceQuiver.MaxHeadSize=0.1;

%Generate RE Total Force Quiver Plot
REForceQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RArmonRForearmFGlobal(:,1),ZTCFQ.RArmonRForearmFGlobal(:,2),ZTCFQ.RArmonRForearmFGlobal(:,3));
REForceQuiver.LineWidth=1;
REForceQuiver.Color=[1 0 0];
REForceQuiver.MaxHeadSize=0.1;
REForceQuiver.AutoScaleFactor=RHForceQuiver.ScaleFactor/REForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Forearm
REMOFLForearmQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RElbowonRForearmMOFGlobal(:,1),ZTCFQ.RElbowonRForearmMOFGlobal(:,2),ZTCFQ.RElbowonRForearmMOFGlobal(:,3));
REMOFLForearmQuiver.LineWidth=1;
REMOFLForearmQuiver.Color=[0 0.75 0];
REMOFLForearmQuiver.MaxHeadSize=0.1;
REMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Right Wrist MOF on Right Forearm
RWristMOFLForearm=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.RWristonRForearmMOFGlobal(:,1),ZTCFQ.RWristonRForearmMOFGlobal(:,2),ZTCFQ.RWristonRForearmMOFGlobal(:,3));
RWristMOFLForearm.LineWidth=1;
RWristMOFLForearm.Color=[0 0.5 0];
RWristMOFLForearm.MaxHeadSize=0.1;
RWristMOFLForearm.AutoScaleFactor=REMOFLForearmQuiver.ScaleFactor/RWristMOFLForearm.ScaleFactor;

%Generate Right Forearm Quivers
RightForearm=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RightForearmdx(:,1),ZTCFQ.RightForearmdy(:,1),ZTCFQ.RightForearmdz(:,1),0);
RightForearm.ShowArrowHead='off';
RightForearm.LineWidth=1;			   
RightForearm.Color=[0 0 0];	

%Add Legend to Plot
legend('','','','','RH Force','RE Force','RE MOF','RH MOF');

%Add a Title
title('Moments of Force Acting on Right Forearm');
subtitle('ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Right Forearm Moments');
pause(PauseTime);

%Close Figure
close(321);

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