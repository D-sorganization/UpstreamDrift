
% Read from model workspace the value in the starting position / velocity
% from the impact optimized model

cd(matlabdrive);
GolfSwing3D
mdlWks=get_param('GolfSwing3D','ModelWorkspace');

% Read from source file
% load("3DModelInputs.mat")
% load("3DModelInputs_TopofBackswing.mat")
load("3DModelInputs_Impact.mat")

% Hip Rotation Position Targets
assignin(mdlWks,"HipStartPositionX",Simulink.Parameter(HipStartPositionX.Value))
assignin(mdlWks,"HipStartPositionY",Simulink.Parameter(HipStartPositionY.Value))
assignin(mdlWks,"HipStartPositionZ",Simulink.Parameter(HipStartPositionZ.Value))

% Hip Rotation Velocity Targets
assignin(mdlWks,"HipStartVelocityX",Simulink.Parameter(HipStartVelocityX.Value))
assignin(mdlWks,"HipStartVelocityY",Simulink.Parameter(HipStartVelocityY.Value))
assignin(mdlWks,"HipStartVelocityZ",Simulink.Parameter(HipStartVelocityZ.Value))

% Hip Translation Position Targets
assignin(mdlWks,"TranslationStartPositionX",Simulink.Parameter(TranslationStartPositionX.Value))
assignin(mdlWks,"TranslationStartPositionY",Simulink.Parameter(TranslationStartPositionY.Value))
assignin(mdlWks,"TranslationStartPositionZ",Simulink.Parameter(TranslationStartPositionZ.Value))

% Hip Translation Velocity Targets
assignin(mdlWks,"TranslationStartVelocityX",Simulink.Parameter(TranslationStartVelocityX.Value))
assignin(mdlWks,"TranslationStartVelocityY",Simulink.Parameter(TranslationStartVelocityY.Value))
assignin(mdlWks,"TranslationStartVelocityZ",Simulink.Parameter(TranslationStartVelocityZ.Value))

% Spine Position Targets
assignin(mdlWks,"SpineStartPositionX",Simulink.Parameter(SpineStartPositionX.Value))
assignin(mdlWks,"SpineStartPositionY",Simulink.Parameter(SpineStartPositionY.Value))

% Spine Velocity Targets
assignin(mdlWks,"SpineStartVelocityX",Simulink.Parameter(SpineStartVelocityX.Value))
assignin(mdlWks,"SpineStartVelocityY",Simulink.Parameter(SpineStartVelocityY.Value))

% Torso Position Targets
assignin(mdlWks,"TorsoStartPosition",Simulink.Parameter(TorsoStartPosition.Value))

% Torso Velocity Targets
assignin(mdlWks,"TorsoStartVelocity",Simulink.Parameter(TorsoStartVelocity.Value))

% LScap Position Targets
assignin(mdlWks,"LScapStartPositionX",Simulink.Parameter(LScapStartPositionX.Value))
assignin(mdlWks,"LScapStartPositionY",Simulink.Parameter(LScapStartPositionY.Value))

% LScap Velocity Targets
assignin(mdlWks,"LScapStartVelocityX",Simulink.Parameter(LScapStartVelocityX.Value))
assignin(mdlWks,"LScapStartVelocityY",Simulink.Parameter(LScapStartVelocityY.Value))

% RScap Position Targets
assignin(mdlWks,"RScapStartPositionX",Simulink.Parameter(RScapStartPositionX.Value))
assignin(mdlWks,"RScapStartPositionY",Simulink.Parameter(RScapStartPositionY.Value))

% RScap Velocity Targets
assignin(mdlWks,"RScapStartVelocityX",Simulink.Parameter(RScapStartVelocityX.Value))
assignin(mdlWks,"RScapStartVelocityY",Simulink.Parameter(RScapStartVelocityY.Value))

% LS Position Targets
assignin(mdlWks,"LSStartPositionX",Simulink.Parameter(LSStartPositionX.Value))
assignin(mdlWks,"LSStartPositionY",Simulink.Parameter(LSStartPositionY.Value))
assignin(mdlWks,"LSStartPositionZ",Simulink.Parameter(LSStartPositionZ.Value))

% LS Velocity Targets
assignin(mdlWks,"LSStartVelocityX",Simulink.Parameter(LSStartVelocityX.Value))
assignin(mdlWks,"LSStartVelocityY",Simulink.Parameter(LSStartVelocityY.Value))
assignin(mdlWks,"LSStartVelocityZ",Simulink.Parameter(LSStartVelocityZ.Value))

% RS Position Targets
assignin(mdlWks,"RSStartPositionX",Simulink.Parameter(RSStartPositionX.Value))
assignin(mdlWks,"RSStartPositionY",Simulink.Parameter(RSStartPositionY.Value))
assignin(mdlWks,"RSStartPositionZ",Simulink.Parameter(RSStartPositionZ.Value))

% RS Velocity Targets
assignin(mdlWks,"RSStartVelocityX",Simulink.Parameter(RSStartVelocityX.Value))
assignin(mdlWks,"RSStartVelocityY",Simulink.Parameter(RSStartVelocityY.Value))
assignin(mdlWks,"RSStartVelocityZ",Simulink.Parameter(RSStartVelocityZ.Value))

% LE Position Targets
assignin(mdlWks,"LEStartPosition",Simulink.Parameter(LEStartPosition.Value))

% LE Velocity Targets
assignin(mdlWks,"LEStartVelocity",Simulink.Parameter(LEStartVelocity.Value))

% RE Position Targets
assignin(mdlWks,"REStartPosition",Simulink.Parameter(REStartPosition.Value))

% RE Velocity Targets
assignin(mdlWks,"REStartVelocity",Simulink.Parameter(REStartVelocity.Value))% RE Position Targets

% LF Position Targets
assignin(mdlWks,"LFStartPosition",Simulink.Parameter(LFStartPosition.Value))

% LF Velocity Targets
assignin(mdlWks,"LFStartVelocity",Simulink.Parameter(LFStartVelocity.Value))

% RF Position Targets
assignin(mdlWks,"RFStartPosition",Simulink.Parameter(RFStartPosition.Value))

% RF Velocity Targets
assignin(mdlWks,"RFStartVelocity",Simulink.Parameter(RFStartVelocity.Value))

% LW Position Targets
assignin(mdlWks,"LWStartPositionX",Simulink.Parameter(LWStartPositionX.Value))
assignin(mdlWks,"LWStartPositionY",Simulink.Parameter(LWStartPositionY.Value))

% LW Velocity Targets
assignin(mdlWks,"LWStartVelocityX",Simulink.Parameter(LWStartVelocityX.Value))
assignin(mdlWks,"LWStartVelocityY",Simulink.Parameter(LWStartVelocityY.Value))

% RW Position Targets
assignin(mdlWks,"RWStartPositionX",Simulink.Parameter(RWStartPositionX.Value))
assignin(mdlWks,"RWStartPositionY",Simulink.Parameter(RWStartPositionY.Value))

% RW Velocity Targets
assignin(mdlWks,"RWStartVelocityX",Simulink.Parameter(RWStartVelocityX.Value))
assignin(mdlWks,"RWStartVelocityY",Simulink.Parameter(RWStartVelocityY.Value))

% Save Model Workspace
% save(mdlWks,'3DModelInputs.mat')

% Clear Workspace
clear
mdlWks=get_param('GolfSwing3D','ModelWorkspace');

% Run Model
GolfSwing3D
