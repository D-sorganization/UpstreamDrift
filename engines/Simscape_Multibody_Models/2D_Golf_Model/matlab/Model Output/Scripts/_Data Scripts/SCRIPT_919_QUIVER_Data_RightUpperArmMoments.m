%Generate Club Quiver Plot
figure(919);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Elbow on Right Upper Arm Total Force Quiver Plot
RElbowQuiver=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RForearmonRArmFGlobal(:,1),Data.RForearmonRArmFGlobal(:,2),Data.RForearmonRArmFGlobal(:,3));
RElbowQuiver.LineWidth=1;
RElbowQuiver.Color=[0 0 1];
RElbowQuiver.AutoScaleFactor=2;
RElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Right Upper Arm Total Force Quiver Plot
RSForceQuiver=quiver3(Data.RSx(:,1),Data.RSy(:,1),Data.RSz(:,1),Data.RSonRArmFGlobal(:,1),Data.RSonRArmFGlobal(:,2),Data.RSonRArmFGlobal(:,3));
RSForceQuiver.LineWidth=1;
RSForceQuiver.Color=[1 0 0];
RSForceQuiver.MaxHeadSize=0.1;
RSForceQuiver.AutoScaleFactor=RElbowQuiver.ScaleFactor/RSForceQuiver.ScaleFactor;

%Generate Right Elbow MOF on Right Upper Arm
REMOFRArmQuiver=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RElbowonRArmMOFGlobal(:,1),Data.RElbowonRArmMOFGlobal(:,2),Data.RElbowonRArmMOFGlobal(:,3));
REMOFRArmQuiver.LineWidth=1;
REMOFRArmQuiver.Color=[0 0.75 0];
REMOFRArmQuiver.MaxHeadSize=0.1;
REMOFRArmQuiver.AutoScaleFactor=2;

%Generate Right Shoulder MOF on Right Upper Arm
RSMOFRArm=quiver3(Data.RSx(:,1),Data.RSy(:,1),Data.RSz(:,1),Data.RShoulderonRArmMOFGlobal(:,1),Data.RShoulderonRArmMOFGlobal(:,2),Data.RShoulderonRArmMOFGlobal(:,3));
RSMOFRArm.LineWidth=1;
RSMOFRArm.Color=[0 0.5 0];
RSMOFRArm.MaxHeadSize=0.1;
RSMOFRArm.AutoScaleFactor=REMOFRArmQuiver.ScaleFactor/RSMOFRArm.ScaleFactor;

%Generate Right Arm Quivers
RightArm=quiver3(Data.RSx(:,1),Data.RSy(:,1),Data.RSz(:,1),Data.RightArmdx(:,1),Data.RightArmdy(:,1),Data.RightArmdz(:,1),0);
RightArm.ShowArrowHead='off';
RightArm.LineWidth=1;			   
RightArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','RE Force','RS Force','RE MOF','RS MOF','');

%Add a Title
title('Moments of Force Acting on Right Upper Arm');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Right Upper Arm Moments');
pause(PauseTime);

%Close Figure
close(919);

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