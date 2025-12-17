%Generate Club Quiver Plot
figure(7);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Elbow on Left Upper Arm Total Force Quiver Plot
LElbowQuiver=quiver3(BASEQ.LEx(:,1),BASEQ.LEy(:,1),BASEQ.LEz(:,1),BASEQ.LForearmonLArmFGlobal(:,1),BASEQ.LForearmonLArmFGlobal(:,2),BASEQ.LForearmonLArmFGlobal(:,3));
LElbowQuiver.LineWidth=1;
LElbowQuiver.Color=[0 0 1];
LElbowQuiver.AutoScaleFactor=2;
LElbowQuiver.MaxHeadSize=0.1;

%Generate Shoulder on Left Upper Arm Total Force Quiver Plot
LSForceQuiver=quiver3(BASEQ.LSx(:,1),BASEQ.LSy(:,1),BASEQ.LSz(:,1),BASEQ.LSonLArmFGlobal(:,1),BASEQ.LSonLArmFGlobal(:,2),BASEQ.LSonLArmFGlobal(:,3));
LSForceQuiver.LineWidth=1;
LSForceQuiver.Color=[1 0 0];
LSForceQuiver.MaxHeadSize=0.1;
LSForceQuiver.AutoScaleFactor=LElbowQuiver.ScaleFactor/LSForceQuiver.ScaleFactor;

%Generate Left Elbow MOF on Left Upper Arm
LEMOFLArmQuiver=quiver3(BASEQ.LEx(:,1),BASEQ.LEy(:,1),BASEQ.LEz(:,1),BASEQ.LElbowonLArmMOFGlobal(:,1),BASEQ.LElbowonLArmMOFGlobal(:,2),BASEQ.LElbowonLArmMOFGlobal(:,3));
LEMOFLArmQuiver.LineWidth=1;
LEMOFLArmQuiver.Color=[0 0.75 0];
LEMOFLArmQuiver.MaxHeadSize=0.1;
LEMOFLArmQuiver.AutoScaleFactor=2;

%Generate Left Shoulder MOF on Left Upper Arm
LSMOFLArm=quiver3(BASEQ.LSx(:,1),BASEQ.LSy(:,1),BASEQ.LSz(:,1),BASEQ.LShoulderonLArmMOFGlobal(:,1),BASEQ.LShoulderonLArmMOFGlobal(:,2),BASEQ.LShoulderonLArmMOFGlobal(:,3));
LSMOFLArm.LineWidth=1;
LSMOFLArm.Color=[0 0.5 0];
LSMOFLArm.MaxHeadSize=0.1;
LSMOFLArm.AutoScaleFactor=LEMOFLArmQuiver.ScaleFactor/LSMOFLArm.ScaleFactor;

%Generate Left Arm Quivers
LeftArm=quiver3(BASEQ.LSx(:,1),BASEQ.LSy(:,1),BASEQ.LSz(:,1),BASEQ.LeftArmdx(:,1),BASEQ.LeftArmdy(:,1),BASEQ.LeftArmdz(:,1),0);
LeftArm.ShowArrowHead='off';		
LeftArm.LineWidth=1;			   
LeftArm.Color=[0 0 0];

%Add Legend to Plot
legend('','','LE Force','LS Force','LE MOF','LS MOF','');

%Add a Title
title('Moments of Force Acting on Left Upper Arm');
subtitle('COMPARISON');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Left Upper Arm Moments');

%Close Figure
close(7);

%Clear Figure from Workspace
clear LElbowQuiver;
clear LSForceQuiver;
clear LEMOFLArmQuiver;
clear LSMOFLArm;
clear LeftArm;
clear RHForceQuiver;
clear Grip;
clear Shaft;