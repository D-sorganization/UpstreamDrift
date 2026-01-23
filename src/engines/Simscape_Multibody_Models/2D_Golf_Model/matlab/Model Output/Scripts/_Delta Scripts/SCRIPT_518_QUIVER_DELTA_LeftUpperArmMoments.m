%Generate Club Quiver Plot
figure(518);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Elbow on Left Upper Arm Total Force Quiver Plot
LElbowQuiver=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),DELTAQ.LForearmonLArmFGlobal(:,1),DELTAQ.LForearmonLArmFGlobal(:,2),DELTAQ.LForearmonLArmFGlobal(:,3));
LElbowQuiver.LineWidth=1;
LElbowQuiver.Color=[0 0 1];
LElbowQuiver.AutoScaleFactor=2;
LElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Left Upper Arm Total Force Quiver Plot
LSForceQuiver=quiver3(ZTCFQ.LSx(:,1),ZTCFQ.LSy(:,1),ZTCFQ.LSz(:,1),DELTAQ.LSonLArmFGlobal(:,1),DELTAQ.LSonLArmFGlobal(:,2),DELTAQ.LSonLArmFGlobal(:,3));
LSForceQuiver.LineWidth=1;
LSForceQuiver.Color=[1 0 0];
LSForceQuiver.MaxHeadSize=0.1;
LSForceQuiver.AutoScaleFactor=LElbowQuiver.ScaleFactor/LSForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Upper Arm
LEMOFLArmQuiver=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),DELTAQ.LElbowonLArmMOFGlobal(:,1),DELTAQ.LElbowonLArmMOFGlobal(:,2),DELTAQ.LElbowonLArmMOFGlobal(:,3));
LEMOFLArmQuiver.LineWidth=1;
LEMOFLArmQuiver.Color=[0 0.75 0];
LEMOFLArmQuiver.MaxHeadSize=0.1;
LEMOFLArmQuiver.AutoScaleFactor=2;

%Generate Left Shoulder MOF on Left Upper Arm
LSMOFLArm=quiver3(ZTCFQ.LSx(:,1),ZTCFQ.LSy(:,1),ZTCFQ.LSz(:,1),DELTAQ.LShoulderonLArmMOFGlobal(:,1),DELTAQ.LShoulderonLArmMOFGlobal(:,2),DELTAQ.LShoulderonLArmMOFGlobal(:,3));
LSMOFLArm.LineWidth=1;
LSMOFLArm.Color=[0 0.5 0];
LSMOFLArm.MaxHeadSize=0.1;
LSMOFLArm.AutoScaleFactor=LEMOFLArmQuiver.ScaleFactor/LSMOFLArm.ScaleFactor;

%Generate Left Arm Quivers
LeftArm=quiver3(ZTCFQ.LSx(:,1),ZTCFQ.LSy(:,1),ZTCFQ.LSz(:,1),DELTAQ.LeftArmdx(:,1),DELTAQ.LeftArmdy(:,1),DELTAQ.LeftArmdz(:,1),0);
LeftArm.ShowArrowHead='off';		
LeftArm.LineWidth=1;			   
LeftArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','','','LE Force','LS Force','LE MOF','LS MOF','');

%Add a Title
title('Moments of Force Acting on Left Upper Arm');
subtitle('DELTA');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Left Upper Arm Moments');
pause(PauseTime);

%Close Figure
close(518);

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