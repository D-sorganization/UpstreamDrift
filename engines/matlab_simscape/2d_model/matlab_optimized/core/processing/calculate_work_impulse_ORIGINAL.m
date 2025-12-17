function [ZTCF_out, DELTA_out] = calculate_work_impulse(ZTCF_in, DELTA_in, config)
% CALCULATE_WORK_IMPULSE - Calculate work and impulse for ZTCF and DELTA tables
%
% Inputs:
%   ZTCF_in - ZTCF table before calculations
%   DELTA_in - DELTA table before calculations
%   config - Configuration structure
%
% Returns:
%   ZTCF_out - ZTCF table with added work/impulse columns
%   DELTA_out - DELTA table with added work/impulse columns
%
% This function replaces SCRIPT_UpdateCalcsforImpulseandWork.m by consolidating
% duplicate calculation logic into a single parameterized function.
%
% Calculations performed:
%   - Linear Work = integral(F · v dt) for each joint
%   - Angular Work = integral(T · ω dt) for each joint
%   - Linear Impulse = integral(F dt) for each joint
%   - Angular Impulse = integral(T dt) for each joint
%
% Author: Optimized Golf Swing Analysis System
% Date: 2025

    if config.verbose
        fprintf('🔬 Calculating work and impulse quantities...\n');
    end

    %% Calculate for ZTCF
    if config.verbose
        fprintf('   Processing ZTCF table...\n');
    end
    ZTCF_out = calculate_single_table(ZTCF_in, config);

    %% Calculate for DELTA
    if config.verbose
        fprintf('   Processing DELTA table...\n');
    end
    DELTA_out = calculate_single_table(DELTA_in, config);

    if config.verbose
        fprintf('✅ Work and impulse calculations complete\n');
        fprintf('   Added %d new columns to each table\n', ...
            width(ZTCF_out) - width(ZTCF_in));
    end

end

function table_out = calculate_single_table(table_in, config)
    % Perform calculations for a single table

    table_out = table_in;
    sample_time = config.base_sample_time;
    num_rows = height(table_in);

    %% Pre-allocate arrays for calculations
    % Forces
    F = zeros(num_rows, 3);
    LHF = zeros(num_rows, 3);
    RHF = zeros(num_rows, 3);
    LEF = zeros(num_rows, 3);
    REF = zeros(num_rows, 3);
    LSF = zeros(num_rows, 3);
    RSF = zeros(num_rows, 3);

    % Torques
    TLS = zeros(num_rows, 3);
    TRS = zeros(num_rows, 3);
    TLE = zeros(num_rows, 3);
    TRE = zeros(num_rows, 3);
    TLW = zeros(num_rows, 3);
    TRW = zeros(num_rows, 3);

    % Sum of moments
    SUMLS = zeros(num_rows, 3);
    SUMRS = zeros(num_rows, 3);
    SUMLE = zeros(num_rows, 3);
    SUMRE = zeros(num_rows, 3);
    SUMLW = zeros(num_rows, 3);
    SUMRW = zeros(num_rows, 3);

    % Velocities
    V = zeros(num_rows, 3);
    LHV = zeros(num_rows, 3);
    RHV = zeros(num_rows, 3);
    LEV = zeros(num_rows, 3);
    REV = zeros(num_rows, 3);
    LSV = zeros(num_rows, 3);
    RSV = zeros(num_rows, 3);

    % Angular velocities
    LSAV = zeros(num_rows, 3);
    RSAV = zeros(num_rows, 3);
    LEAV = zeros(num_rows, 3);
    REAV = zeros(num_rows, 3);
    LWAV = zeros(num_rows, 3);
    RWAV = zeros(num_rows, 3);

    %% Extract data from table
    for i = 1:num_rows
        % Forces
        F(i,:) = table_in{i, "TotalHandForceGlobal"};
        LHF(i,:) = table_in{i, "LWonClubFGlobal"};
        RHF(i,:) = table_in{i, "RWonClubFGlobal"};
        LEF(i,:) = table_in{i, "LArmonLForearmFGlobal"};
        REF(i,:) = table_in{i, "RArmonRForearmFGlobal"};
        LSF(i,:) = table_in{i, "LSonLArmFGlobal"};
        RSF(i,:) = table_in{i, "RSonRArmFGlobal"};

        % Sum of moments
        SUMLS(i,:) = table_in{i, "SumofMomentsLSonLArm"};
        SUMRS(i,:) = table_in{i, "SumofMomentsRSonRArm"};
        SUMLE(i,:) = table_in{i, "SumofMomentsLEonLForearm"};
        SUMRE(i,:) = table_in{i, "SumofMomentsREonRForearm"};
        SUMLW(i,:) = table_in{i, "SumofMomentsLWristonClub"};
        SUMRW(i,:) = table_in{i, "SumofMomentsRWristonClub"};

        % Torques
        TLS(i,:) = table_in{i, "LSonLArmTGlobal"};
        TRS(i,:) = table_in{i, "RSonRArmTGlobal"};
        TLE(i,:) = table_in{i, "LArmonLForearmTGlobal"};
        TRE(i,:) = table_in{i, "RArmonRForearmTGlobal"};
        TLW(i,:) = table_in{i, "LWonClubTGlobal"};
        TRW(i,:) = table_in{i, "RWonClubTGlobal"};

        % Linear velocities
        V(i,:) = table_in{i, "MidHandVelocity"};
        LHV(i,:) = table_in{i, "LeftHandVelocity"};
        RHV(i,:) = table_in{i, "RightHandVelocity"};
        LEV(i,:) = table_in{i, "LEvGlobal"};
        REV(i,:) = table_in{i, "REvGlobal"};
        LSV(i,:) = table_in{i, "LSvGlobal"};
        RSV(i,:) = table_in{i, "RSvGlobal"};

        % Angular velocities
        LSAV(i,:) = table_in{i, "LSvGlobal"};
        RSAV(i,:) = table_in{i, "RSvGlobal"};
        LEAV(i,:) = table_in{i, "LEvGlobal"};
        REAV(i,:) = table_in{i, "REvGlobal"};
        LWAV(i,:) = table_in{i, "LeftWristGlobalAV"};
        RWAV(i,:) = table_in{i, "RightWristGlobalAV"};
    end

    %% Calculate power (dot products)
    % Linear power
    P = dot(F, V, 2);
    LHP = dot(LHF, LHV, 2);
    RHP = dot(RHF, RHV, 2);
    LEP = dot(LEF, LEV, 2);
    REP = dot(REF, REV, 2);
    LSP = dot(LSF, LSV, 2);
    RSP = dot(RSF, RSV, 2);

    % Angular power
    LSAP = dot(TLS, LSAV, 2);
    RSAP = dot(TRS, RSAV, 2);
    LEAP = dot(TLE, LEAV, 2);
    REAP = dot(TRE, REAV, 2);
    LWAP = dot(TLW, LWAV, 2);
    RWAP = dot(TRW, RWAV, 2);

    %% Integrate to get work (cumulative trapezoid integration)
    % Linear work
    table_out.LinearWorkonClub = cumtrapz(table_in.Time, P);
    table_out.LHLinearWorkonClub = cumtrapz(table_in.Time, LHP);
    table_out.RHLinearWorkonClub = cumtrapz(table_in.Time, RHP);
    table_out.LELinearWorkonForearm = cumtrapz(table_in.Time, LEP);
    table_out.RELinearWorkonForearm = cumtrapz(table_in.Time, REP);
    table_out.LSLinearWorkonArm = cumtrapz(table_in.Time, LSP);
    table_out.RSLinearWorkonArm = cumtrapz(table_in.Time, RSP);

    % Angular work
    table_out.LSAngularWorkonArm = cumtrapz(table_in.Time, LSAP);
    table_out.RSAngularWorkonArm = cumtrapz(table_in.Time, RSAP);
    table_out.LEAngularWorkonForearm = cumtrapz(table_in.Time, LEAP);
    table_out.REAngularWorkonForearm = cumtrapz(table_in.Time, REAP);
    table_out.LWAngularWorkonClub = cumtrapz(table_in.Time, LWAP);
    table_out.RWAngularWorkonClub = cumtrapz(table_in.Time, RWAP);

    %% Calculate impulse (integrate forces/torques)
    % Linear impulse (integrate each component separately, then combine)
    for dim = 1:3
        F_impulse(:,dim) = cumtrapz(table_in.Time, F(:,dim));
        LHF_impulse(:,dim) = cumtrapz(table_in.Time, LHF(:,dim));
        RHF_impulse(:,dim) = cumtrapz(table_in.Time, RHF(:,dim));
        LEF_impulse(:,dim) = cumtrapz(table_in.Time, LEF(:,dim));
        REF_impulse(:,dim) = cumtrapz(table_in.Time, REF(:,dim));
        LSF_impulse(:,dim) = cumtrapz(table_in.Time, LSF(:,dim));
        RSF_impulse(:,dim) = cumtrapz(table_in.Time, RSF(:,dim));
    end

    % Store impulse vectors as cell arrays (each row is a 1x3 vector)
    table_out.TotalHandLinearImpulse = num2cell(F_impulse, 2);
    table_out.LHLinearImpulse = num2cell(LHF_impulse, 2);
    table_out.RHLinearImpulse = num2cell(RHF_impulse, 2);
    table_out.LELinearImpulse = num2cell(LEF_impulse, 2);
    table_out.RELinearImpulse = num2cell(REF_impulse, 2);
    table_out.LSLinearImpulse = num2cell(LSF_impulse, 2);
    table_out.RSLinearImpulse = num2cell(RSF_impulse, 2);

    % Angular impulse (integrate torques and sum of moments)
    for dim = 1:3
        TLS_impulse(:,dim) = cumtrapz(table_in.Time, TLS(:,dim));
        TRS_impulse(:,dim) = cumtrapz(table_in.Time, TRS(:,dim));
        TLE_impulse(:,dim) = cumtrapz(table_in.Time, TLE(:,dim));
        TRE_impulse(:,dim) = cumtrapz(table_in.Time, TRE(:,dim));
        TLW_impulse(:,dim) = cumtrapz(table_in.Time, TLW(:,dim));
        TRW_impulse(:,dim) = cumtrapz(table_in.Time, TRW(:,dim));

        SUMLS_impulse(:,dim) = cumtrapz(table_in.Time, SUMLS(:,dim));
        SUMRS_impulse(:,dim) = cumtrapz(table_in.Time, SUMRS(:,dim));
        SUMLE_impulse(:,dim) = cumtrapz(table_in.Time, SUMLE(:,dim));
        SUMRE_impulse(:,dim) = cumtrapz(table_in.Time, SUMRE(:,dim));
        SUMLW_impulse(:,dim) = cumtrapz(table_in.Time, SUMLW(:,dim));
        SUMRW_impulse(:,dim) = cumtrapz(table_in.Time, SUMRW(:,dim));
    end

    % Store angular impulse vectors
    table_out.LSAngularImpulse = num2cell(TLS_impulse, 2);
    table_out.RSAngularImpulse = num2cell(TRS_impulse, 2);
    table_out.LEAngularImpulse = num2cell(TLE_impulse, 2);
    table_out.REAngularImpulse = num2cell(TRE_impulse, 2);
    table_out.LWAngularImpulse = num2cell(TLW_impulse, 2);
    table_out.RWAngularImpulse = num2cell(TRW_impulse, 2);

    % Store sum of moments impulse
    table_out.SUMLSAngularImpulse = num2cell(SUMLS_impulse, 2);
    table_out.SUMRSAngularImpulse = num2cell(SUMRS_impulse, 2);
    table_out.SUMLEAngularImpulse = num2cell(SUMLE_impulse, 2);
    table_out.SUMREAngularImpulse = num2cell(SUMRE_impulse, 2);
    table_out.SUMLWAngularImpulse = num2cell(SUMLW_impulse, 2);
    table_out.SUMRWAngularImpulse = num2cell(SUMRW_impulse, 2);

end
