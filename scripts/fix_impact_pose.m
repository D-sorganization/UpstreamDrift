% scripts/fix_impact_pose.m
%
% Write a corrected 3DModelInputs_Impact_fixed.mat alongside the existing
% Impact MAT. The values below are a CREDIBLE STARTER impact pose:
% pelvis open ~45 deg to target, forward spine tilt 28 deg, lead arm
% extended, trail elbow lightly flexed, lead wrist flat.
%
% This script is intentionally simple. It does NOT replace the existing
% file; instead it writes a sibling _fixed.mat the user can adopt by
% renaming.
%
% Usage:
%   cd src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/inputs
%   run fix_impact_pose.m
%
% After running, swap the files in
% SCRIPT_TransferStartPositionVelocityIntoModelFromMATFile.m
% from "3DModelInputs_Impact.mat" to "3DModelInputs_Impact_fixed.mat"
% and run the model.

deg = @(v) Simulink.Parameter(v);

%% Pelvis
HipStartPositionX = deg(0);
HipStartPositionY = deg(0);
HipStartPositionZ = deg(-45);     % open hips at impact (preserved)
HipStartVelocityX = deg(0);
HipStartVelocityY = deg(0);
HipStartVelocityZ = deg(-50);

TranslationStartPositionX = deg(0);
TranslationStartPositionY = deg(0);
TranslationStartPositionZ = deg(0);
TranslationStartVelocityX = deg(0);
TranslationStartVelocityY = deg(0);
TranslationStartVelocityZ = deg(0);

%% Spine — forward tilt is the load-bearing fix
SpineStartPositionX = deg(28);    % WAS 0 — added forward tilt
SpineStartPositionY = deg(-5);    % WAS 0 — small lead-side bend
SpineStartVelocityX = deg(0);
SpineStartVelocityY = deg(0);

%% Torso axial — keep ~half the impact rotation
TorsoStartPosition = deg(-30);    % WAS -45 (top-of-backswing magnitude)
TorsoStartVelocity = deg(-50);

%% Scapulae — modest values
LScapStartPositionX = deg(15);    % WAS 34.16
LScapStartPositionY = deg(0);
LScapStartVelocityX = deg(0);
LScapStartVelocityY = deg(0);
RScapStartPositionX = deg(5);
RScapStartPositionY = deg(5);
RScapStartVelocityX = deg(0);
RScapStartVelocityY = deg(0);

%% Shoulders — credible impact, arms extended toward ball
LSStartPositionX = deg(-30);
LSStartPositionY = deg(-10);
LSStartPositionZ = deg(-45);      % WAS -135.72 (top-of-backswing)
LSStartVelocityX = deg(0);
LSStartVelocityY = deg(0);
LSStartVelocityZ = deg(0);

RSStartPositionX = deg(-30);
RSStartPositionY = deg(0);
RSStartPositionZ = deg(30);       % WAS 96.03
RSStartVelocityX = deg(0);
RSStartVelocityY = deg(0);
RSStartVelocityZ = deg(0);

%% Elbows
LEStartPosition = deg(10);        % lead arm nearly straight
LEStartVelocity = deg(0);
REStartPosition = deg(35);        % WAS 100.70 (top-of-backswing)
REStartVelocity = deg(0);

%% Forearms
LFStartPosition = deg(15);
LFStartVelocity = deg(0);
RFStartPosition = deg(0);
RFStartVelocity = deg(0);

%% Wrists — flat lead wrist at impact
LWStartPositionX = deg(0);        % WAS -97.84 (fully cocked)
LWStartPositionY = deg(5);
LWStartVelocityX = deg(0);
LWStartVelocityY = deg(0);
RWStartPositionX = deg(0);        % WAS -80.02
RWStartPositionY = deg(-5);
RWStartVelocityX = deg(0);
RWStartVelocityY = deg(0);

save('3DModelInputs_Impact_fixed.mat', ...
    'HipStartPositionX','HipStartPositionY','HipStartPositionZ', ...
    'HipStartVelocityX','HipStartVelocityY','HipStartVelocityZ', ...
    'TranslationStartPositionX','TranslationStartPositionY','TranslationStartPositionZ', ...
    'TranslationStartVelocityX','TranslationStartVelocityY','TranslationStartVelocityZ', ...
    'SpineStartPositionX','SpineStartPositionY','SpineStartVelocityX','SpineStartVelocityY', ...
    'TorsoStartPosition','TorsoStartVelocity', ...
    'LScapStartPositionX','LScapStartPositionY','LScapStartVelocityX','LScapStartVelocityY', ...
    'RScapStartPositionX','RScapStartPositionY','RScapStartVelocityX','RScapStartVelocityY', ...
    'LSStartPositionX','LSStartPositionY','LSStartPositionZ', ...
    'LSStartVelocityX','LSStartVelocityY','LSStartVelocityZ', ...
    'RSStartPositionX','RSStartPositionY','RSStartPositionZ', ...
    'RSStartVelocityX','RSStartVelocityY','RSStartVelocityZ', ...
    'LEStartPosition','LEStartVelocity','REStartPosition','REStartVelocity', ...
    'LFStartPosition','LFStartVelocity','RFStartPosition','RFStartVelocity', ...
    'LWStartPositionX','LWStartPositionY','LWStartVelocityX','LWStartVelocityY', ...
    'RWStartPositionX','RWStartPositionY','RWStartVelocityX','RWStartVelocityY');

fprintf('Wrote 3DModelInputs_Impact_fixed.mat\n');
