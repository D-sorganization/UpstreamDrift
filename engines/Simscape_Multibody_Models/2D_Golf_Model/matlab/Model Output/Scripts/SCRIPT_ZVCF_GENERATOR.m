% ZVCF Data Generation Script
% 
% This script generates the data for a scenario in which you take the
% applied joint torques at any given point in the swing and apply them to a
% golfer in the same static pose. "Zero Velocity Counterfactual". This can
% be done with and without gravity on by changing the GolfSwingZVCF model
% solver settings. The reason for computing without gravity is that the
% ZVCF adds up to the Delta (the difference between the base swing and
% ZTCF) when gravity isn't counted twice. As the system is a series of
% linear differential equations, the principle of superposition applies and
% the net effect of all actions on the system is additive. In the ZTCF
% gravity is included as one of the passive contributors (along with shaft
% flex and momentum). In principle, the ZVCF is the effect of the joint
% torques on the interaction forces everywhere in the swing.

% The ZVCF is calculated by importing starting positions / pose of the
% model and assigning all velocities to be zero. The joint torques are
% applied as constant values and the joint interaction forces are
% calculated at time zero and tabulated. 

cd(matlabdrive);
cd '2DModel';
warning off Simulink:cgxe:LeakedJITEngine;

% Copy the model inputs file that was used to generate the ZTCF and run the
% GolfSwing model previously. Write it to the ZVCF scripts folder
% temporarily.
copyfile ModelInputs.mat 'Scripts'/'_ZVCF Scripts'/;
cd 'Scripts/_ZVCF Scripts/';
% Rename the file using the movefile function
movefile 'ModelInputs.mat' 'ModelInputsZVCF.mat';
% Move the file using the copyfile function
cd(matlabdrive);
cd '2DModel';
copyfile Scripts/'_ZVCF Scripts'/'ModelInputsZVCF.mat';

% Delete the file that was copied into the ZVCF Scripts folder
cd(matlabdrive);
cd '2DModel/Scripts/';
cd '_ZVCF Scripts';
delete 'ModelInputsZVCF.mat';


% Go back to the main folder. Open GolfSwingZVCF model. The model is set to
% look for ModelInputsZVCF when it is opened.
cd(matlabdrive);
cd '2DModel';
GolfSwingZVCF

% Load mdlWks for ZVCF Model from File
cd(matlabdrive);
cd '2DModel';
mdlWks=get_param('GolfSwingZVCF','ModelWorkspace');
mdlWks.DataSource = 'MAT-File';
mdlWks.FileName = 'ModelInputsZVCF.mat';
mdlWks.reload;

% Set up the model simulation parameters.
set_param(GolfSwingZVCF,"ReturnWorkspaceOutputs","on");
set_param(GolfSwingZVCF,"FastRestart","off");
set_param(GolfSwingZVCF,"MaxStep","0.001");

%Turn off the warning that a directory already exists when you create it.
warning('off', 'MATLAB:MKDIR:DirectoryExists');
warning off Simulink:Masking:NonTunableParameterChangedDuringSimulation;
warning off Simulink:Engine:NonTunableVarChangedInFastRestart;
warning off Simulink:Engine:NonTunableVarChangedMaxWarnings;

%Set the killswitch time to 1 second so it doesn't ever trigger
assignin(mdlWks,'KillswitchStepTime',Simulink.Parameter(1));
assignin(mdlWks,'StopTime',Simulink.Parameter(0.05));

% Run the model to generate data table. Save model output as "out".
out=sim("GolfSwingZVCF");

% Run Table Generation Script on "out"
cd(matlabdrive);
cd '2DModel/Scripts';
SCRIPT_TableGeneration;

% Copy Data to ZTCF Table to Get Variable Names
ZVCFTable=Data; %Create a table called ZTCFTable from Data.
ZVCFTable(:,:)=[]; %Delete All Data in ZTCF Table and Replace with Blanks 

%Now we have a table with all of the right columns and variables that can
%be written to from the output of the ZVCF model when it generates data.
%The script runs up to this point and does what it needs to do.
%
% The general approach will be to provide the model with input positions
% and joint torques. Then the model will be simulated and the values from
% time zero will be copied into the ZVCF table generated to receive data
% above. The only issue is that all times will be zero. The time will then
% be generated using the step time data from the loop once the row has been
% copied. 

% Begin Generation of ZVCF Data by Looping

    %Pick i and j based on number of data points to run ZVCF at.
    %for i=0:280
    for i=0:28

    %Scale counter to match desired times
    %j=i/1000;
    j=i/100;

    %Display Percentage
    %ZVCFPercentComplete=i/280*100
    ZVCFPercentComplete=i/28*100

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

    % Run Table Generation Script on "out"
    SCRIPT_TableGeneration;

    % Write the first row in the Data table to the ZVCFTable
    CopyRow=Data(1,:);
    CopyRow{1,1}=j;
    ZVCFTable=[ZVCFTable;CopyRow];

    end

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the Q Table By Running Script
cd(matlabdrive);
cd '2DModel/Scripts/_ZVCF Scripts';
SCRIPT_ZVCF_QTableGenerate;
cd(matlabdrive);
cd '2DModel';


    
    
    
   