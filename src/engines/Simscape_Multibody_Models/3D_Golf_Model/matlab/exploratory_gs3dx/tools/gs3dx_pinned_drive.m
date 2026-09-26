function [vars, block_parameters] = gs3dx_pinned_drive(info, mdl)
%GS3DX_PINNED_DRIVE  Impact drive with the right shoulder's start state pinned.
%
%   [VARS, BLOCK_PARAMETERS] = GS3DX_PINNED_DRIVE(INFO, MDL) returns
%   GS3DX_SIMULATE overrides for MDL: the impact drive (GS3DX_DRIVE) with
%   the right shoulder's start angles and rates set to the state the
%   original model assembles at t = 0, and all six of its target priorities
%   raised from Low to High.
%
%   Why: both arms hold the club, so the initial assembly cannot meet the
%   right shoulder's Low-priority targets and settles on a compromise.  A
%   Gimbal Joint compromises per axis and a Spherical Joint on one rotation,
%   so GS3DX_Quat and GS3DX_Slim start from different states under the
%   plain impact drive.  Pinned, both start from the same state (#10955).
%
%   The original's assembled state is read from the stored impact baseline
%   (GS3DX_BASELINE_FILE); no model is modified.

    arguments
        info (1,1) struct
        mdl (1,:) char
    end
    S = load(gs3dx_baseline_file(info, "impact"));
    flat = S.baseline.flat;
    vars = gs3dx_drive(info, "impact", mdl);
    blk = [mdl '/Right Shoulder Joint/Gimbal Joint'];
    block_parameters = cell(0, 3);
    position = struct('X', "AngularPositionX", 'Y', "AngularPositionY", 'Z', "AngularPosition_Z");
    for axis = 'XYZ'
        vars.(['RSStartPosition' axis]) = local_start(flat, "RSLogs/" + position.(axis));
        vars.(['RSStartVelocity' axis]) = local_start(flat, "RSLogs/AngularVelocity" + axis);
        block_parameters(end+1, :) = {blk, ['R' lower(axis) 'PositionTargetPriority'], 'High'}; %#ok<AGROW>
        block_parameters(end+1, :) = {blk, ['R' lower(axis) 'VelocityTargetPriority'], 'High'}; %#ok<AGROW>
    end
end

function v = local_start(flat, name)
    k = find(flat.names == name);
    assert(isscalar(k), 'gs3dx:pinned', 'Baseline has no single signal %s', name);
    v = flat.data{k}(1, 1);
end
