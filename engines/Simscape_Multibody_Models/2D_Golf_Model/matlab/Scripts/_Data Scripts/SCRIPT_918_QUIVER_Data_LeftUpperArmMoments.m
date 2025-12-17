%Generate Club Quiver Plot
figure(918);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Elbow on Left Upper Arm Total Force Quiver Plot
LElbowQuiver=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LForearmonLArmFGlobal(:,1),Data.LForearmonLArmFGlobal(:,2),Data.LForearmonLArmFGlobal(:,3));
LElbowQuiver.LineWidth=1;
LElbowQuiver.Color=[0 0 1];
LElbowQuiver.AutoScaleFactor=2;
LElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Left Upper Arm Total Force Quiver Plot
LSForceQuiver=quiver3(Data.LSx(:,1),Data.LSy(:,1),Data.LSz(:,1),Data.LSonLArmFGlobal(:,1),Data.LSonLArmFGlobal(:,2),Data.LSonLArmFGlobal(:,3));
LSForceQuiver.LineWidth=1;
LSForceQuiver.Color=[1 0 0];
LSForceQuiver.MaxHeadSize=0.1;
LSForceQuiver.AutoScaleFactor=LElbowQuiver.ScaleFactor/LSForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Upper Arm
LEMOFLArmQuiver=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LElbowonLArmMOFGlobal(:,1),Data.LElbowonLArmMOFGlobal(:,2),Data.LElbowonLArmMOFGlobal(:,3));
LEMOFLArmQuiver.LineWidth=1;
LEMOFLArmQuiver.Color=[0 0.75 0];
LEMOFLArmQuiver.MaxHeadSize=0.1;
LEMOFLArmQuiver.AutoScaleFactor=2;

%Generate Left Shoulder MOF on Left Upper Arm
LSMOFLArm=quiver3(Data.LSx(:,1),Data.LSy(:,1),Data.LSz(:,1),Data.LShoulderonLArmMOFGlobal(:,1),Data.LShoulderonLArmMOFGlobal(:,2),Data.LShoulderonLArmMOFGlobal(:,3));
LSMOFLArm.LineWidth=1;
LSMOFLArm.Color=[0 0.5 0];
LSMOFLArm.MaxHeadSize=0.1;
LSMOFLArm.AutoScaleFactor=LEMOFLArmQuiver.ScaleFactor/LSMOFLArm.ScaleFactor;

%Generate Left Arm Quivers
LeftArm=quiver3(Data.LSx(:,1),Data.LSy(:,1),Data.LSz(:,1),Data.LeftArmdx(:,1),Data.LeftArmdy(:,1),Data.LeftArmdz(:,1),0);
LeftArm.ShowArrowHead='off';		
LeftArm.LineWidth=1;			   
LeftArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','LE Force','LS Force','LE MOF','LS MOF','');

%Add a Title
title('Moments of Force Acting on Left Upper Arm');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Left Upper Arm Moments');
pause(PauseTime);

%Close Figure
close(918);

%Clear Figure from Workspace
clear LElbowQuiver;
clear LSForceQuiver;
clear LEMOFLArmQuiver;
clear LSMOFLArm;
clear LeftArm;
clear RHForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;