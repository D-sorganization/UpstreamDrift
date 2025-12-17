% This file is designed to iteratively control the 3D golf swing model. It
% will set target controls and then measure the deviation in each angular
% position and velocityat the time of interest defined in this script.
% Based on this a correction will be made to the control parameters and the
% model will be rerun for a desired number of learning loops.

% Define number of iterative tuning corrections to run:
i=5;

cd(matlabdrive);
GolfSwing3D;
%out=sim(GolfSwing3D);

% Load the desired impact targets into the model
SCRIPT_LoadImpactStateTargets;
% Load the Model Into Starting Position Kinematics
SCRIPT_LoadTopofBackswing;

for i=i:n
    % Run the Model
    out=sim(GolfSwing3D);
    % Generate data table and then generate the table of TargetDeltas
    cd(matlabdrive);
    cd 'Scripts/_Model Data Scripts';
    SCRIPT_3D_TableGeneration;
    SCRIPT_Data_3D_TotalWorkandPowerCalculation;
    SCRIPT_Data_3D_CHPandMPPCalculation;
    SCRIPT_Data_3D_TableofValues;
    cd(matlabdrive);
    SCRIPT_ComputeControlDeltas;
    SCRIPT_TargetDeltaAdjustmentFactorComputation;
end

% Rerun the script without making adjustments to the model workspace
% correction values so that the tuning parameters are kept.

out=sim(GolfSwing3D);
cd(matlabdrive);
cd 'Scripts/_Model Data Scripts';
SCRIPT_3D_TableGeneration;
% SCRIPT_Data_3D_TotalWorkandPowerCalculation;
% SCRIPT_Data_3D_CHPandMPPCalculation;
% SCRIPT_Data_3D_TableofValues;
% cd(matlabdrive);
% SCRIPT_ComputeControlDeltas;

clear i;
