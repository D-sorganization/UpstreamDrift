function [ZTCFQ_updated, DELTAQ_updated] = calculateWorkAndImpulse3D(ZTCFQ, DELTAQ)
% CALCULATEWORKANDIMPULSE3D Numerically computes work and impulse for segments.
%   [ZTCFQ_updated, DELTAQ_updated] = CALCULATEWORKANDIMPULSE3D(ZTCFQ, DELTAQ)
%   Calculates linear work (integral of Power = F.V), linear impulse (integral of F),
%   angular work (integral of Angular Power = T.AV), and angular impulse
%   (integral of Sum of Moments) for various body segments based on the
%   data in the ZTCFQ and DELTAQ tables. New columns for these calculated
%   values are added to the output tables.
%
%   This function is the converted version of SCRIPT_UpdateCalcsforImpulseandWork_3D.m.
%   It operates on tables expected to be on a uniform time grid (like the 'Q' tables).
%
%   Input:
%       ZTCFQ  - Table containing ZTCF data on a uniform time grid,
%                including columns for forces, torques, velocities,
%                angular velocities, and sum of moments.
%       DELTAQ - Table containing Delta data on the same uniform time grid,
%                including columns for forces, torques, velocities,
%                angular velocities, and sum of moments.
%
%   Output:
%       ZTCFQ_updated  - Updated ZTCFQ table with added work and impulse columns.
%       DELTAQ_updated - Updated DELTAQ table with added work and impulse columns.

% Assign input tables to internal variables for modification
ZTCFQ_updated = ZTCFQ;
DELTAQ_updated = DELTAQ;

% --- Get Time Step (Assuming Uniform Grid) ---
% Calculate the time step from the input table's time vector.
% Since the tables are on a uniform TsQ grid, the difference between
% any two consecutive time points should be the time step.
if height(ZTCFQ_updated) > 1
    TsQ = ZTCFQ_updated.Time(2) - ZTCFQ_updated.Time(1);
else
    % Handle case with 0 or 1 data point (cannot calculate impulse/work)
    warning('Input table has less than 2 rows. Cannot calculate work/impulse.');
    % Return the original tables without adding columns
    return;
end

% --- Process ZTCFQ Table ---

fprintf('Calculating work and impulse for ZTCFQ...\n');

% Extract necessary data columns from ZTCFQ
% Ensure these column names match your logged signals from the 3D model
try
    % Forces (Global frame, expected as Nx3)
    F_ZTCF = ZTCFQ_updated{:, ["TotalHandForceGlobalX", "TotalHandForceGlobalY", "TotalHandForceGlobalZ"]};
    LHF_ZTCF = ZTCFQ_updated{:, ["LHonClubFGlobalX", "LHonClubFGlobalY", "LHonClubFGlobalZ"]};
    RHF_ZTCF = ZTCFQ_updated{:, ["RHonClubFGlobalX", "RHonClubFGlobalY", "RHonClubFGlobalZ"]};
    LEF_ZTCF = ZTCFQ_updated{:, ["LArmonLForearmFGlobalX", "LArmonLForearmFGlobalY", "LArmonLForearmFGlobalZ"]};
    REF_ZTCF = ZTCFQ_updated{:, ["RArmonRForearmFGlobalX", "RArmonRForearmFGlobalY", "RArmonRForearmFGlobalZ"]};
    LSF_ZTCF = ZTCFQ_updated{:, ["LSonLArmFGlobalX", "LSonLArmFGlobalY", "LSonLArmFGlobalZ"]};
    RSF_ZTCF = ZTCFQ_updated{:, ["RSonRArmFGlobalX", "RSonRArmFGlobalY", "RSonRArmFGlobalZ"]};

    % Sum of Moments (Global frame, expected as Nx3) - Used for Angular Impulse
    SUMLS_ZTCF = ZTCFQ_updated{:, ["SumofMomentsLSonLArmX", "SumofMomentsLSonLArmY", "SumofMomentsLSonLArmZ"]};
    SUMRS_ZTCF = ZTCFQ_updated{:, ["SumofMomentsRSonRArmX", "SumofMomentsRSonRArmY", "SumofMomentsRSonRArmZ"]};
    SUMLE_ZTCF = ZTCFQ_updated{:, ["SumofMomentsLEonLForearmX", "SumofMomentsLEonLForearmY", "SumofMomentsLEonLForearmZ"]};
    SUMRE_ZTCF = ZTCFQ_updated{:, ["SumofMomentsREonRForearmX", "SumofMomentsREonRForearmY", "SumofMomentsREonRForearmZ"]};
    % Assuming LH/RH naming for wrist moments
    SUMLH_ZTCF = ZTCFQ_updated{:, ["SumofMomentsLHonClubX", "SumofMomentsLHonClubY", "SumofMomentsLHonClubZ"]};
    SUMRH_ZTCF = ZTCFQ_updated{:, ["SumofMomentsRHonClubX", "SumofMomentsRHonClubY", "SumofMomentsRHonClubZ"]};

    % Torques (Global frame, expected as Nx3) - Used for Angular Work
    TLS_ZTCF = ZTCFQ_updated{:, ["LSonLArmTGlobalX", "LSonLArmTGlobalY", "LSonLArmTGlobalZ"]};
    TRS_ZTCF = ZTCFQ_updated{:, ["RSonRArmTGlobalX", "RSonRArmTGlobalY", "RSonRArmTGlobalZ"]};
    TLE_ZTCF = ZTCFQ_updated{:, ["LArmonLForearmTGlobalX", "LArmonLForearmTGlobalY", "LArmonLForearmTGlobalZ"]};
    TRE_ZTCF = ZTCFQ_updated{:, ["RArmonRForearmTGlobalX", "RArmonRForearmTGlobalY", "RArmonRForearmTGlobalZ"]};
    % Assuming LH/RH naming for wrist torques
    TLH_ZTCF = ZTCFQ_updated{:, ["LHonClubTGlobalX", "LHonClubTGlobalY", "LHonClubTGlobalZ"]};
    TRH_ZTCF = ZTCFQ_updated{:, ["RHonClubTGlobalX", "RHonClubTGlobalY", "RHonClubTGlobalZ"]};

    % Velocities (Global frame, expected as Nx3)
    V_ZTCF = ZTCFQ_updated{:, ["MidHandVelocityX", "MidHandVelocityY", "MidHandVelocityZ"]};
    LHV_ZTCF = ZTCFQ_updated{:, ["LeftHandVelocityX", "LeftHandVelocityY", "LeftHandVelocityZ"]};
    RHV_ZTCF = ZTCFQ_updated{:, ["RightHandVelocityX", "RightHandVelocityY", "RightHandVelocityZ"]};
    LEV_ZTCF = ZTCFQ_updated{:, ["LEvGlobalX", "LEvGlobalY", "LEvGlobalZ"]};
    REV_ZTCF = ZTCFQ_updated{:, ["REvGlobalX", "REvGlobalY", "REvGlobalZ"]};
    LSV_ZTCF = ZTCFQ_updated{:, ["LSvGlobalX", "LSvGlobalY", "LSvGlobalZ"]};
    RSV_ZTCF = ZTCFQ_updated{:, ["RSvGlobalX", "RSvGlobalY", "RSvGlobalZ"]};

    % Angular Velocities (Global frame, expected as Nx3)
    LSAV_ZTCF = ZTCFQ_updated{:, ["LSAVGlobalX", "LSAVGlobalY", "LSAVGlobalZ"]};
    RSAV_ZTCF = ZTCFQ_updated{:, ["RSAVGlobalX", "RSAVGlobalY", "RSAVGlobalZ"]};
    LEAV_ZTCF = ZTCFQ_updated{:, ["LEAVGlobalX", "LEAVGlobalY", "LEAVGlobalZ"]};
    REAV_ZTCF = ZTCFQ_updated{:, ["REAVGlobalX", "REAVGlobalY", "REAVGlobalZ"]};
    LHAV_ZTCF = ZTCFQ_updated{:, ["LeftWristGlobalAVX", "LeftWristGlobalAVY", "LeftWristGlobalAVZ"]};
    RHAV_ZTCF = ZTCFQ_updated{:, ["RightWristGlobalAVX", "RightWristGlobalAVY", "RightWristGlobalAVZ"]};

catch ME
    warning('Error extracting required columns for ZTCFQ calculations: %s', ME.message);
    % Return the original tables as is if columns are missing
    ZTCFQ_updated = ZTCFQ;
    DELTAQ_updated = DELTAQ;
    return; % Exit the function
end


% Calculate Linear Power (Dot product of Force and Linear Velocity)
P_ZTCF = sum(F_ZTCF .* V_ZTCF, 2); % Element-wise product then sum across columns
LHP_ZTCF = sum(LHF_ZTCF .* LHV_ZTCF, 2);
RHP_ZTCF = sum(RHF_ZTCF .* RHV_ZTCF, 2);
LEP_ZTCF = sum(LEF_ZTCF .* LEV_ZTCF, 2);
REP_ZTCF = sum(REF_ZTCF .* REV_ZTCF, 2);
LSP_ZTCF = sum(LSF_ZTCF .* LSV_ZTCF, 2);
RSP_ZTCF = sum(RSF_ZTCF .* RSV_ZTCF, 2);

% Calculate Angular Power (Dot product of Torque and Angular Velocity)
LSAP_ZTCF = sum(TLS_ZTCF .* LSAV_ZTCF, 2);
RSAP_ZTCF = sum(TRS_ZTCF .* RSAV_ZTCF, 2);
LEAP_ZTCF = sum(TLE_ZTCF .* LEAV_ZTCF, 2);
REAP_ZTCF = sum(TRE_ZTCF .* REAV_ZTCF, 2);
LHAP_ZTCF = sum(TLH_ZTCF .* LHAV_ZTCF, 2);
RHAP_ZTCF = sum(TRH_ZTCF .* RHAV_ZTCF, 2);


% Numerically Integrate to find Work and Impulse (using cumtrapz)
% cumtrapz(X, Y) integrates Y with respect to X. Using Time is more robust.
timeVec = ZTCFQ_updated.Time;

% Linear Work (Integral of Linear Power)
ZTCFQ_updated.("LinearWorkonClub") = cumtrapz(timeVec, P_ZTCF);
ZTCFQ_updated.("LHLinearWorkonClub") = cumtrapz(timeVec, LHP_ZTCF);
ZTCFQ_updated.("RHLinearWorkonClub") = cumtrapz(timeVec, RHP_ZTCF);
ZTCFQ_updated.("LELinearWorkonForearm") = cumtrapz(timeVec, LEP_ZTCF);
ZTCFQ_updated.("RELinearWorkonForearm") = cumtrapz(timeVec, REP_ZTCF);
ZTCFQ_updated.("LSLinearWorkonArm") = cumtrapz(timeVec, LSP_ZTCF);
ZTCFQ_updated.("RSLinearWorkonArm") = cumtrapz(timeVec, RSP_ZTCF);

% Linear Impulse (Integral of Force) - Note: cumtrapz on Nx3 matrix integrates each column
ZTCFQ_updated.("LinearImpulseonClub") = cumtrapz(timeVec, F_ZTCF);
ZTCFQ_updated.("LHLinearImpulseonClub") = cumtrapz(timeVec, LHF_ZTCF);
ZTCFQ_updated.("RHLinearImpulseonClub") = cumtrapz(timeVec, RHF_ZTCF);
ZTCFQ_updated.("LELinearImpulseonForearm") = cumtrapz(timeVec, LEF_ZTCF);
ZTCFQ_updated.("RELinearImpulseonForearm") = cumtrapz(timeVec, REF_ZTCF);
ZTCFQ_updated.("LSLinearImpulseonArm") = cumtrapz(timeVec, LSF_ZTCF);
ZTCFQ_updated.("RSLinearImpulseonArm") = cumtrapz(timeVec, RSF_ZTCF);

% Angular Impulse (Integral of Sum of Moments)
ZTCFQ_updated.("LSAngularImpulseonArm") = cumtrapz(timeVec, SUMLS_ZTCF);
ZTCFQ_updated.("RSAngularImpulseonArm") = cumtrapz(timeVec, SUMRS_ZTCF);
ZTCFQ_updated.("LEAngularImpulseonForearm") = cumtrapz(timeVec, SUMLE_ZTCF);
ZTCFQ_updated.("REAngularImpulseonForearm") = cumtrapz(timeVec, SUMRE_ZTCF);
ZTCFQ_updated.("LHAngularImpulseonClub") = cumtrapz(timeVec, SUMLH_ZTCF);
ZTCFQ_updated.("RHAngularImpulseonClub") = cumtrapz(timeVec, SUMRH_ZTCF);

% Angular Work (Integral of Angular Power)
ZTCFQ_updated.("LSAngularWorkonArm") = cumtrapz(timeVec, LSAP_ZTCF);
ZTCFQ_updated.("RSAngularWorkonArm") = cumtrapz(timeVec, RSAP_ZTCF);
ZTCFQ_updated.("LEAngularWorkonForearm") = cumtrapz(timeVec, LEAP_ZTCF);
ZTCFQ_updated.("REAngularWorkonForearm") = cumtrapz(timeVec, REAP_ZTCF);
ZTCFQ_updated.("LHAngularWorkonClub") = cumtrapz(timeVec, LHAP_ZTCF);
ZTCFQ_updated.("RHAngularWorkonClub") = cumtrapz(timeVec, RHAP_ZTCF);

fprintf('ZTCFQ work and impulse calculations complete.\n');

% --- Process DELTAQ Table ---

fprintf('Calculating work and impulse for DELTAQ...\n');

% Extract necessary data columns from DELTAQ
% Ensure these column names match your logged signals
try
    % Forces (Global frame, expected as Nx3)
    F_DELTA = DELTAQ_updated{:, ["TotalHandForceGlobalX", "TotalHandForceGlobalY", "TotalHandForceGlobalZ"]};
    LHF_DELTA = DELTAQ_updated{:, ["LHonClubFGlobalX", "LHonClubFGlobalY", "LHonClubFGlobalZ"]};
    RHF_DELTA = DELTAQ_updated{:, ["RHonClubFGlobalX", "RHonClubFGlobalY", "RHonClubFGlobalZ"]};
    LEF_DELTA = DELTAQ_updated{:, ["LArmonLForearmFGlobalX", "LArmonLForearmFGlobalY", "LArmonLForearmFGlobalZ"]};
    REF_DELTA = DELTAQ_updated{:, ["RArmonRForearmFGlobalX", "RArmonRForearmFGlobalY", "RArmonRForearmFGlobalZ"]};
    LSF_DELTA = DELTAQ_updated{:, ["LSonLArmFGlobalX", "LSonLArmFGlobalY", "LSonLArmFGlobalZ"]};
    RSF_DELTA = DELTAQ_updated{:, ["RSonRArmFGlobalX", "RSonRArmFGlobalY", "RSonRArmFGlobalZ"]};

    % Sum of Moments (Global frame, expected as Nx3) - Used for Angular Impulse
    SUMLS_DELTA = DELTAQ_updated{:, ["SumofMomentsLSonLArmX", "SumofMomentsLSonLArmY", "SumofMomentsLSonLArmZ"]};
    SUMRS_DELTA = DELTAQ_updated{:, ["SumofMomentsRSonRArmX", "SumofMomentsRSonRArmY", "SumofMomentsRSonRArmZ"]};
    SUMLE_DELTA = DELTAQ_updated{:, ["SumofMomentsLEonLForearmX", "SumofMomentsLEonLForearmY", "SumofMomentsLEonLForearmZ"]};
    SUMRE_DELTA = DELTAQ_updated{:, ["SumofMomentsREonRForearmX", "SumofMomentsREonRForearmY", "SumofMomentsREonRForearmZ"]};
    SUMLH_DELTA = DELTAQ_updated{:, ["SumofMomentsLHonClubX", "SumofMomentsLHonClubY", "SumofMomentsLHonClubZ"]};
    SUMRH_DELTA = DELTAQ_updated{:, ["SumofMomentsRHonClubX", "SumofMomentsRHonClubY", "SumofMomentsRHonClubZ"]};

    % Torques (Global frame, expected as Nx3) - Used for Angular Work
    TLS_DELTA = DELTAQ_updated{:, ["LSonLArmTGlobalX", "LSonLArmTGlobalY", "LSonLArmTGlobalZ"]};
    TRS_DELTA = DELTAQ_updated{:, ["RSonRArmTGlobalX", "RSonRArmTGlobalY", "RSonRArmTGlobalZ"]};
    TLE_DELTA = DELTAQ_updated{:, ["LArmonLForearmTGlobalX", "LArmonLForearmTGlobalY", "LArmonLForearmTGlobalZ"]};
    TRE_DELTA = DELTAQ_updated{:, ["RArmonRForearmTGlobalX", "RArmonRForearmTGlobalY", "RArmonRForearmTGlobalZ"]};
    TLH_DELTA = DELTAQ_updated{:, ["LHonClubTGlobalX", "LHonClubTGlobalY", "LHonClubTGlobalZ"]};
    TRH_DELTA = DELTAQ_updated{:, ["RHonClubTGlobalX", "RHonClubTGlobalY", "RHonClubTGlobalZ"]};

    % Velocities (Global frame, expected as Nx3) - NOTE: Using ZTCFQ velocities as in original script
    % This is a potential point of confusion/review. If DELTA represents the *difference*
    % in forces/torques, should it be multiplied by BASE velocities or DELTA velocities?
    % Sticking to original script's logic using ZTCFQ velocities for now.
    V_DELTA = ZTCFQ{:, ["MidHandVelocityX", "MidHandVelocityY", "MidHandVelocityZ"]};
    LHV_DELTA = ZTCFQ{:, ["LeftHandVelocityX", "LeftHandVelocityY", "LeftHandVelocityZ"]};
    RHV_DELTA = ZTCFQ{:, ["RightHandVelocityX", "RightHandVelocityY", "RightHandVelocityZ"]};
    LEV_DELTA = ZTCFQ{:, ["LEvGlobalX", "LEvGlobalY", "LEvGlobalZ"]};
    REV_DELTA = ZTCFQ{:, ["REvGlobalX", "REvGlobalY", "REvGlobalZ"]};
    LSV_DELTA = ZTCFQ{:, ["LSvGlobalX", "LSvGlobalY", "LSvGlobalZ"]};
    RSV_DELTA = ZTCFQ{:, ["RSvGlobalX", "RSvGlobalY", "RSvGlobalZ"]};

    % Angular Velocities (Global frame, expected as Nx3) - NOTE: Using ZTCFQ angular velocities as in original script
    LSAV_DELTA = ZTCFQ{:, ["LSAVGlobalX", "LSAVGlobalY", "LSAVGlobalZ"]};
    RSAV_DELTA = ZTCFQ{:, ["RSAVGlobalX", "RSAVGlobalY", "RSAVGlobalZ"]};
    LEAV_DELTA = ZTCFQ{:, ["LEAVGlobalX", "LEAVGlobalY", "LEAVGlobalZ"]};
    REAV_DELTA = ZTCFQ{:, ["REAVGlobalX", "REAVGlobalY", "REAVGlobalZ"]};
    LHAV_DELTA = ZTCFQ{:, ["LeftWristGlobalAVX", "LeftWristGlobalAVY", "LeftWristGlobalAVZ"]};
    RHAV_DELTA = ZTCFQ{:, ["RightWristGlobalAVX", "RightWristGlobalAVY", "RightWristGlobalAVZ"]};

catch ME
    warning('Error extracting required columns for DELTAQ calculations: %s', ME.message);
    % Return the original tables as is if columns are missing
    ZTCFQ_updated = ZTCFQ;
    DELTAQ_updated = DELTAQ;
    return; % Exit the function
end

% Calculate Linear Power (Dot product of Force and Linear Velocity)
P_DELTA = sum(F_DELTA .* V_DELTA, 2); % Element-wise product then sum across columns
LHP_DELTA = sum(LHF_DELTA .* LHV_DELTA, 2);
RHP_DELTA = sum(RHF_DELTA .* RHV_DELTA, 2);
LEP_DELTA = sum(LEF_DELTA .* LEV_DELTA, 2);
REP_DELTA = sum(REF_DELTA .* REV_DELTA, 2);
LSP_DELTA = sum(LSF_DELTA .* LSV_DELTA, 2);
RSP_DELTA = sum(RSF_DELTA .* RSV_DELTA, 2);

% Calculate Angular Power (Dot product of Torque and Angular Velocity)
LSAP_DELTA = sum(TLS_DELTA .* LSAV_DELTA, 2);
RSAP_DELTA = sum(TRS_DELTA .* RSAV_DELTA, 2);
LEAP_DELTA = sum(TLE_DELTA .* LEAV_DELTA, 2);
REAP_DELTA = sum(TRE_DELTA .* REAV_DELTA, 2);
LHAP_DELTA = sum(TLH_DELTA .* LHAV_DELTA, 2);
RHAP_DELTA = sum(TRH_DELTA .* RHAV_DELTA, 2);


% Numerically Integrate to find Work and Impulse (using cumtrapz)
% Use the time vector from DELTAQ (should be the same as ZTCFQ)
timeVec = DELTAQ_updated.Time;

% Linear Work (Integral of Linear Power)
DELTAQ_updated.("LinearWorkonClub") = cumtrapz(timeVec, P_DELTA);
DELTAQ_updated.("LHLinearWorkonClub") = cumtrapz(timeVec, LHP_DELTA);
DELTAQ_updated.("RHLinearWorkonClub") = cumtrapz(timeVec, RHP_DELTA);
DELTAQ_updated.("LELinearWorkonForearm") = cumtrapz(timeVec, LEP_DELTA);
DELTAQ_updated.("RELinearWorkonForearm") = cumtrapz(timeVec, REP_DELTA);
DELTAQ_updated.("LSLinearWorkonArm") = cumtrapz(timeVec, LSP_DELTA);
DELTAQ_updated.("RSLinearWorkonArm") = cumtrapz(timeVec, RSP_DELTA);

% Linear Impulse (Integral of Force)
DELTAQ_updated.("LinearImpulseonClub") = cumtrapz(timeVec, F_DELTA);
DELTAQ_updated.("LHLinearImpulseonClub") = cumtrapz(timeVec, LHF_DELTA);
DELTAQ_updated.("RHLinearImpulseonClub") = cumtrapz(timeVec, RHF_DELTA);
DELTAQ_updated.("LELinearImpulseonForearm") = cumtrapz(timeVec, LEF_DELTA);
DELTAQ_updated.("RELinearImpulseonForearm") = cumtrapz(timeVec, REF_DELTA);
DELTAQ_updated.("LSLinearImpulseonArm") = cumtrapz(timeVec, LSF_DELTA);
DELTAQ_updated.("RSLinearImpulseonArm") = cumtrapz(timeVec, RSF_DELTA);

% Angular Impulse (Integral of Sum of Moments)
DELTAQ_updated.("LSAngularImpulseonArm") = cumtrapz(timeVec, SUMLS_DELTA);
DELTAQ_updated.("RSAngularImpulseonArm") = cumtrapz(timeVec, SUMRS_DELTA);
DELTAQ_updated.("LEAngularImpulseonForearm") = cumtrapz(timeVec, SUMLE_DELTA);
DELTAQ_updated.("REAngularImpulseonForearm") = cumtrapz(timeVec, SUMRE_DELTA);
DELTAQ_updated.("LHAngularImpulseonClub") = cumtrapz(timeVec, SUMLH_DELTA);
DELTAQ_updated.("RHAngularImpulseonClub") = cumtrapz(timeVec, SUMRH_DELTA);

% Angular Work (Integral of Angular Power)
DELTAQ_updated.("LSAngularWorkonArm") = cumtrapz(timeVec, LSAP_DELTA);
DELTAQ_updated.("RSAngularWorkonArm") = cumtrapz(timeVec, RSAP_DELTA);
DELTAQ_updated.("LEAngularWorkonForearm") = cumtrapz(timeVec, LEAP_DELTA);
DELTAQ_updated.("REAngularWorkonForearm") = cumtrapz(timeVec, REAP_DELTA);
DELTAQ_updated.("LHAngularWorkonClub") = cumtrapz(timeVec, LHAP_DELTA);
DELTAQ_updated.("RHAngularWorkonClub") = cumtrapz(timeVec, RHAP_DELTA);

fprintf('DELTAQ work and impulse calculations complete.\n');

% No need for explicit 'clear' statements within a function.
% All intermediate variables are local and cleared automatically.

end
