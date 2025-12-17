%ZVCF Solo Script - Pick a time and run the ZVCF at that point to create a
%visualization of the effort the golfer makes.

GolfSwingZVCF;
j=0.20;
%Set the killswitch time to 1 second so it doesn't ever trigger
assignin(mdlWks,'KillswitchStepTime',Simulink.Parameter(1));
assignin(mdlWks,'StopTime',Simulink.Parameter(0.05));

% Read the joint torque values at the counter time
HipJointTorque=interp1(BASE.Time,BASE.JointTorqueHip,j,'linear');
TorsoJointTorque=interp1(BASE.Time,BASE.JointTorqueTorso,j,'linear');
LScapJointTorque=interp1(BASE.Time,BASE.JointTorqueLScap,j,'linear');
RScapJointTorque=interp1(BASE.Time,BASE.JointTorqueRScap,j,'linear');
LShoulderJointTorque=interp1(BASE.Time,BASE.JointTorqueLShoulder,j,'linear');
RShoulderJointTorque=interp1(BASE.Time,BASE.JointTorqueRShoulder,j,'linear');
LElbowJointTorque=interp1(BASE.Time,BASE.JointTorqueLElbow,j,'linear');
RElbowJointTorque=interp1(BASE.Time,BASE.JointTorqueRElbow,j,'linear');
LWristJointTorque=interp1(BASE.Time,BASE.JointTorqueLWrist,j,'linear');
RWristJointTorque=interp1(BASE.Time,BASE.JointTorqueRWrist,j,'linear');

%Read the position values at the counter time and convert to degrees
HipPosition=interp1(BASE.Time,BASE.HipPosition,j,'linear')*180/pi;
TorsoPosition=interp1(BASE.Time,BASE.TorsoPosition,j,'linear')*180/pi;
LScapPosition=interp1(BASE.Time,BASE.LScapPosition,j,'linear')*180/pi;
RScapPosition=interp1(BASE.Time,BASE.RScapPosition,j,'linear')*180/pi;
LShoulderPosition=interp1(BASE.Time,BASE.LSPosition,j,'linear')*180/pi;
RShoulderPosition=interp1(BASE.Time,BASE.RSPosition,j,'linear')*180/pi;
LElbowPosition=interp1(BASE.Time,BASE.LeftElbowPosition,j,'linear')*180/pi;
RElbowPosition=interp1(BASE.Time,BASE.RightElbowPosition,j,'linear')*180/pi;
LWristPosition=interp1(BASE.Time,BASE.LeftWristPosition,j,'linear')*180/pi;

% Assign in the torque values to the model workspace
assignin(mdlWks,'JointTorqueHip',Simulink.Parameter(HipJointTorque));
assignin(mdlWks,'JointTorqueTorso',Simulink.Parameter(TorsoJointTorque));
assignin(mdlWks,'JointTorqueLScap',Simulink.Parameter(LScapJointTorque));
assignin(mdlWks,'JointTorqueRScap',Simulink.Parameter(RScapJointTorque));
assignin(mdlWks,'JointTorqueLShoulder',Simulink.Parameter(LShoulderJointTorque));
assignin(mdlWks,'JointTorqueRShoulder',Simulink.Parameter(RShoulderJointTorque));
assignin(mdlWks,'JointTorqueLElbow',Simulink.Parameter(LElbowJointTorque));
assignin(mdlWks,'JointTorqueRElbow',Simulink.Parameter(RElbowJointTorque));
assignin(mdlWks,'JointTorqueLWrist',Simulink.Parameter(LWristJointTorque));
assignin(mdlWks,'JointTorqueRWrist',Simulink.Parameter(RWristJointTorque));

% Assign in position and velocity values to the model workspace
assignin(mdlWks,'HipStartPosition',Simulink.Parameter(HipPosition));
assignin(mdlWks,'HipStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'TorsoStartPosition',Simulink.Parameter(TorsoPosition));
assignin(mdlWks,'TorsoStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'LScapStartPosition',Simulink.Parameter(LScapPosition));
assignin(mdlWks,'LScapStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'RScapStartPosition',Simulink.Parameter(RScapPosition));
assignin(mdlWks,'RScapStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'LShoulderStartPosition',Simulink.Parameter(LShoulderPosition));
assignin(mdlWks,'LShoudlerStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'RShoulderStartPosition',Simulink.Parameter(RShoulderPosition));
assignin(mdlWks,'RShoulderStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'LElbowStartPosition',Simulink.Parameter(LElbowPosition));
assignin(mdlWks,'LElbowStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'RElbowStartPosition',Simulink.Parameter(RElbowPosition));
assignin(mdlWks,'RElbowStartVelocity',Simulink.Parameter(0));
assignin(mdlWks,'LWristStartPosition',Simulink.Parameter(LWristPosition));
assignin(mdlWks,'LWristStartVelocity',Simulink.Parameter(0));

% Run the model to generate data table. Save model output as "out".
out=sim("GolfSwingZVCF");

clear i;
clear j;
clear HipJointTorque;
clear TorsoJointTorque;
clear LScapJointTorque;
clear RScapJointTorque;
clear LShoulderJointTorque;
clear RShoulderJointTorque;
clear LElbowJointTorque;
clear RElbowJointTorque;
clear LWristJointTorque;
clear RWristJointTorque;
clear HipPosition;
clear TorsoPosition;
clear LScapPosition;
clear RScapPosition;
clear LShoulderPosition;
clear RShoulderPosition;
clear LElbowPosition;
clear RElbowPosition;
clear LWristPosition;
clear RWristPosition;
clear ZVCFPercentComplete;
clear CopyRow;
clear out;

%SCRIPT_TableGeneration;
