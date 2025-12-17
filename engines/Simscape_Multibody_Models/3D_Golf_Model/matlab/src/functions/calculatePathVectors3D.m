function [BASEQ, ZTCFQ, DELTAQ] = calculatePathVectors3D(BASEQ, ZTCFQ, DELTAQ)
% CALCULATEPATHVECTORS3D Computes 3D path vectors for Clubhead and Midpoint Hand.
%   [BASEQ, ZTCFQ, DELTAQ] = CALCULATEPATHVECTORS3D(BASEQ, ZTCFQ, DELTAQ)
%   calculates the 3D displacement vectors between consecutive time steps
%   for the Clubhead (CH) and Midpoint Hand (MP) based on their position
%   coordinates in the input tables. The calculated vectors (CHPx, CHPy,
%   CHPz, MPPx, MPPy, MPPz) are added as new columns to the respective tables.
%
%   This function operates on tables that are expected to be on a uniform
%   time grid, typically the 'Q' tables (BASEQ, ZTCFQ, DELTAQ).
%
%   Input:
%       BASEQ  - Table containing Base data on a uniform time grid,
%                including columns 'CHx', 'CHy', 'CHz', 'MPx', 'MPy', 'MPz'.
%       ZTCFQ  - Table containing ZTCF data on the same uniform time grid,
%                including columns 'CHx', 'CHy', 'CHz', 'MPx', 'MPy', 'MPz'.
%       DELTAQ - Table containing Delta data on the same uniform time grid,
%                including columns 'CHx', 'CHy', 'CHz', 'MPx', 'MPy', 'MPz'.
%
%   Output:
%       BASEQ  - Updated BASEQ table with added path vector columns.
%       ZTCFQ  - Updated ZTCFQ table with added path vector columns.
%       DELTAQ - Updated DELTAQ table with added path vector columns.

    arguments
        BASEQ table
        ZTCFQ table
        DELTAQ table
    end

% --- Compute Clubhead Path Vectors (CHPx, CHPy, CHPz) ---
% Calculated as the displacement from one time step to the next.
% The original script assigned the difference (t+dt - t) to time t.
% It also copied the second-to-last difference to the last row.
% We replicate this logic using vectorized operations.

% Calculate differences using diff. This results in a vector of size N-1.
diffCHx_BASEQ = diff(BASEQ.CHx);
diffCHy_BASEQ = diff(BASEQ.CHy);
diffCHz_BASEQ = diff(BASEQ.CHz);

diffCHx_ZTCFQ = diff(ZTCFQ.CHx);
diffCHy_ZTCFQ = diff(ZTCFQ.CHy);
diffCHz_ZTCFQ = diff(ZTCFQ.CHz);

diffCHx_DELTAQ = diff(DELTAQ.CHx);
diffCHy_DELTAQ = diff(DELTAQ.CHy);
diffCHz_DELTAQ = diff(DELTAQ.CHz);

% Replicate the original script's logic: copy the last difference to the final row.
% This makes the resulting vector the same length as the original table.
BASEQ.("CHPx") = [diffCHx_BASEQ; diffCHx_BASEQ(end)];
BASEQ.("CHPy") = [diffCHy_BASEQ; diffCHy_BASEQ(end)];
BASEQ.("CHPz") = [diffCHz_BASEQ; diffCHz_BASEQ(end)];

ZTCFQ.("CHPx") = [diffCHx_ZTCFQ; diffCHx_ZTCFQ(end)];
ZTCFQ.("CHPy") = [diffCHy_ZTCFQ; diffCHy_ZTCFQ(end)];
ZTCFQ.("CHPz") = [diffCHz_ZTCFQ; diffCHz_ZTCFQ(end)];

DELTAQ.("CHPx") = [diffCHx_DELTAQ; diffCHx_DELTAQ(end)];
DELTAQ.("CHPy") = [diffCHy_DELTAQ; diffCHy_DELTAQ(end)];
DELTAQ.("CHPz") = [diffCHz_DELTAQ; diffCHz_DELTAQ(end)];


% --- Compute Midpoint Hand Path Vectors (MPPx, MPPy, MPPz) ---
% Calculated as the displacement from one time step to the next, similar to CH.

% Calculate differences using diff.
diffMPx_BASEQ = diff(BASEQ.MPx);
diffMPy_BASEQ = diff(BASEQ.MPy);
diffMPz_BASEQ = diff(BASEQ.MPz);

diffMPx_ZTCFQ = diff(ZTCFQ.MPx);
diffMPy_ZTCFQ = diff(ZTCFQ.MPy);
diffMPz_ZTCFQ = diff(ZTCFQ.MPz);

diffMPx_DELTAQ = diff(DELTAQ.MPx);
diffMPy_DELTAQ = diff(DELTAQ.MPy);
diffMPz_DELTAQ = diff(DELTAQ.MPz);

% Replicate the original script's logic: copy the last difference to the final row.
BASEQ.("MPPx") = [diffMPx_BASEQ; diffMPx_BASEQ(end)];
BASEQ.("MPPy") = [diffMPy_BASEQ; diffMPy_BASEQ(end)];
BASEQ.("MPPz") = [diffMPz_BASEQ; diffMPz_BASEQ(end)];

ZTCFQ.("MPPx") = [diffMPx_ZTCFQ; diffMPx_ZTCFQ(end)];
ZTCFQ.("MPPy") = [diffMPy_ZTCFQ; diffMPy_ZTCFQ(end)];
ZTCFQ.("MPPz") = [diffMPz_ZTCFQ; diffMPz_ZTCFQ(end)];

DELTAQ.("MPPx") = [diffMPx_DELTAQ; diffMPx_DELTAQ(end)];
DELTAQ.("MPPy") = [diffMPy_DELTAQ; diffMPy_DELTAQ(end)];
DELTAQ.("MPPz") = [diffMPz_DELTAQ; diffMPz_DELTAQ(end)];

% No need for explicit 'clear' statements within a function.
% Temporary variables like diffCHx_BASEQ are local and will be cleared automatically.

end
