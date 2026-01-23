function PLOT_ComparisonQuiver(BASEQ, DELTAQ)
%PLOT_ComparisonQuiverSuite - Visual comparison of ZVCF and DELTA forces
% Mirrors MASTER_SCRIPT_ComparisonCharts_3D.m

%% Plot 1: ZVCF vs DELTA LHRH Forces
figure(1);
PZL = [BASEQ.LHx, BASEQ.LHy, BASEQ.LHz];
PZR = [BASEQ.RHx, BASEQ.RHy, BASEQ.RHz];
PZM = [BASEQ.MPx, BASEQ.MPy, BASEQ.MPz];

FZL = BASEQ.LHonClubFGlobal;
FZR = BASEQ.RHonClubFGlobal;
FZT = BASEQ.TotalHandForceGlobal;

FDL = DELTAQ.LHonClubFGlobal;
FDR = DELTAQ.RHonClubFGlobal;
FDT = DELTAQ.TotalHandForceGlobal;

overlayQuiverForces({PZL, PZR, PZM, PZL, PZR, PZM}, ...
                    {FZL, FZR, FZT, FDL, FDR, FDT}, ...
    'Tags', {'ZVCF LH Force','ZVCF RH Force','ZVCF Net Force', ...
             'DELTA LH Force','DELTA RH Force','DELTA Net Force'}, ...
    'Colors', {[0 0 1], [1 0 0], [0 1 0], [0 0 0.5], [0.5 0 0], [0 0.5 0]}, ...
    'Title', 'LHRH Forces: ZVCF vs DELTA');
savefig('Comparison Quiver Plots/Comparison_Quiver Plot - ZVCF vs DELTA LHRH Forces');

%% Add other comparison plots as needed here

end
