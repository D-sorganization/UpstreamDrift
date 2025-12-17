%Generate Club Quiver Plot
figure(519);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Elbow on Right Upper Arm Total Force Quiver Plot
RElbowQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),DELTAQ.RForearmonRArmFGlobal(:,1),DELTAQ.RForearmonRArmFGlobal(:,2),DELTAQ.RForearmonRArmFGlobal(:,3));
RElbowQuiver.LineWidth=1;
RElbowQuiver.Color=[0 0 1];
RElbowQuiver.AutoScaleFactor=2;
RElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Right Upper Arm Total Force Quiver Plot
RSForceQuiver=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),DELTAQ.RSonRArmFGlobal(:,1),DELTAQ.RSonRArmFGlobal(:,2),DELTAQ.RSonRArmFGlobal(:,3));
RSForceQuiver.LineWidth=1;
RSForceQuiver.Color=[1 0 0];
RSForceQuiver.MaxHeadSize=0.1;
RSForceQuiver.AutoScaleFactor=RElbowQuiver.ScaleFactor/RSForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Upper Arm
REMOFRArmQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),DELTAQ.RElbowonRArmMOFGlobal(:,1),DELTAQ.RElbowonRArmMOFGlobal(:,2),DELTAQ.RElbowonRArmMOFGlobal(:,3));
REMOFRArmQuiver.LineWidth=1;
REMOFRArmQuiver.Color=[0 0.75 0];
REMOFRArmQuiver.MaxHeadSize=0.1;
REMOFRArmQuiver.AutoScaleFactor=2;

%Generate Right Shoulder MOF on Right Upper Arm
RSMOFRArm=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),DELTAQ.RShoulderonRArmMOFGlobal(:,1),DELTAQ.RShoulderonRArmMOFGlobal(:,2),DELTAQ.RShoulderonRArmMOFGlobal(:,3));
RSMOFRArm.LineWidth=1;
RSMOFRArm.Color=[0 0.5 0];
RSMOFRArm.MaxHeadSize=0.1;
RSMOFRArm.AutoScaleFactor=REMOFRArmQuiver.ScaleFactor/RSMOFRArm.ScaleFactor;

%Generate Right Arm Quivers
RightArm=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),DELTAQ.RightArmdx(:,1),DELTAQ.RightArmdy(:,1),DELTAQ.RightArmdz(:,1),0);
RightArm.ShowArrowHead='off';
RightArm.LineWidth=1;			   
RightArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','RE Force','RS Force','RE MOF','RS MOF','');

%Add a Title
title('Moments of Force Acting on Right Upper Arm');
subtitle('DELTA');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Right Upper Arm Moments');
pause(PauseTime);

%Close Figure
close(519);

%Clear Figure from Workspace
clear RElbowQuiver;
clear RSForceQuiver;
clear REMOFRArmQuiver;
clear RSMOFRArm;
clear RightArm;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;