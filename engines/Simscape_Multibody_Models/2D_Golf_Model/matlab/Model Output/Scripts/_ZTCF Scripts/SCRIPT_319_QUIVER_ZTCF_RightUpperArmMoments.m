%Generate Club Quiver Plot
figure(319);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Elbow on Right Upper Arm Total Force Quiver Plot
RElbowQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RForearmonRArmFGlobal(:,1),ZTCFQ.RForearmonRArmFGlobal(:,2),ZTCFQ.RForearmonRArmFGlobal(:,3));
RElbowQuiver.LineWidth=1;
RElbowQuiver.Color=[0 0 1];
RElbowQuiver.AutoScaleFactor=2;
RElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Right Upper Arm Total Force Quiver Plot
RSForceQuiver=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),ZTCFQ.RSonRArmFGlobal(:,1),ZTCFQ.RSonRArmFGlobal(:,2),ZTCFQ.RSonRArmFGlobal(:,3));
RSForceQuiver.LineWidth=1;
RSForceQuiver.Color=[1 0 0];
RSForceQuiver.MaxHeadSize=0.1;
RSForceQuiver.AutoScaleFactor=RElbowQuiver.ScaleFactor/RSForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Upper Arm
REMOFRArmQuiver=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RElbowonRArmMOFGlobal(:,1),ZTCFQ.RElbowonRArmMOFGlobal(:,2),ZTCFQ.RElbowonRArmMOFGlobal(:,3));
REMOFRArmQuiver.LineWidth=1;
REMOFRArmQuiver.Color=[0 0.75 0];
REMOFRArmQuiver.MaxHeadSize=0.1;
REMOFRArmQuiver.AutoScaleFactor=2;

%Generate Right Shoulder MOF on Right Upper Arm
RSMOFRArm=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),ZTCFQ.RShoulderonRArmMOFGlobal(:,1),ZTCFQ.RShoulderonRArmMOFGlobal(:,2),ZTCFQ.RShoulderonRArmMOFGlobal(:,3));
RSMOFRArm.LineWidth=1;
RSMOFRArm.Color=[0 0.5 0];
RSMOFRArm.MaxHeadSize=0.1;
RSMOFRArm.AutoScaleFactor=REMOFRArmQuiver.ScaleFactor/RSMOFRArm.ScaleFactor;

%Generate Right Arm Quivers
RightArm=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),ZTCFQ.RightArmdx(:,1),ZTCFQ.RightArmdy(:,1),ZTCFQ.RightArmdz(:,1),0);
RightArm.ShowArrowHead='off';
RightArm.LineWidth=1;			   
RightArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','RE Force','RS Force','RE MOF','RS MOF','');

%Add a Title
title('Moments of Force Acting on Right Upper Arm');
subtitle('ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Right Upper Arm Moments');
pause(PauseTime);

%Close Figure
close(319);

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