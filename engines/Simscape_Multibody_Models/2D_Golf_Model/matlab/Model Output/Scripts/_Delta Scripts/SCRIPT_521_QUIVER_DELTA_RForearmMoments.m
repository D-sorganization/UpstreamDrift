%Generate Club Quiver Plot
figure(521);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate RW Total Force Quiver Plot
RHForceQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),DELTAQ.ClubonRWFGlobal(:,1),DELTAQ.ClubonRWFGlobal(:,2),DELTAQ.ClubonRWFGlobal(:,3));
RHForceQuiver.LineWidth=1;
RHForceQuiver.Color=[0 0 1];
RHForceQuiver.AutoScaleFactor=2;
RHForceQuiver.MaxHeadSize=0.1;

%Generate RE Total Force Quiver Plot
REForceQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),DELTAQ.RArmonRForearmFGlobal(:,1),DELTAQ.RArmonRForearmFGlobal(:,2),DELTAQ.RArmonRForearmFGlobal(:,3));
REForceQuiver.LineWidth=1;
REForceQuiver.Color=[1 0 0];
REForceQuiver.MaxHeadSize=0.1;
REForceQuiver.AutoScaleFactor=RHForceQuiver.ScaleFactor/REForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Forearm
REMOFLForearmQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),DELTAQ.RElbowonRForearmMOFGlobal(:,1),DELTAQ.RElbowonRForearmMOFGlobal(:,2),DELTAQ.RElbowonRForearmMOFGlobal(:,3));
REMOFLForearmQuiver.LineWidth=1;
REMOFLForearmQuiver.Color=[0 0.75 0];
REMOFLForearmQuiver.MaxHeadSize=0.1;
REMOFLForearmQuiver.AutoScaleFactor=2;

%Generate Right Wrist MOF on Right Forearm
RWristMOFLForearm=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),DELTAQ.RWristonRForearmMOFGlobal(:,1),DELTAQ.RWristonRForearmMOFGlobal(:,2),DELTAQ.RWristonRForearmMOFGlobal(:,3));
RWristMOFLForearm.LineWidth=1;
RWristMOFLForearm.Color=[0 0.5 0];
RWristMOFLForearm.MaxHeadSize=0.1;
RWristMOFLForearm.AutoScaleFactor=REMOFLForearmQuiver.ScaleFactor/RWristMOFLForearm.ScaleFactor;

%Generate Right Forearm Quivers
RightForearm=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),DELTAQ.RightForearmdx(:,1),DELTAQ.RightForearmdy(:,1),DELTAQ.RightForearmdz(:,1),0);
RightForearm.ShowArrowHead='off';
RightForearm.LineWidth=1;			   
RightForearm.Color=[0 0 0];	

%Add Legend to Plot
legend('','','','','RH Force','RE Force','RE MOF','RH MOF');

%Add a Title
title('Moments of Force Acting on Right Forearm');
subtitle('DELTA');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Right Forearm Moments');
pause(PauseTime);

%Close Figure
close(521);

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