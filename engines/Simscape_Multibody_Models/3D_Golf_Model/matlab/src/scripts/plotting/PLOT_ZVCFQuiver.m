function PLOT_ZVCFQuiver(ZVCFQ, DELTAQ, BASEQ)
%PLOT_ZVCFQuiver - Replicates all quiver plots from MASTER_SCRIPT_ZVCF_CHARTS_3D.m

%% Plot 1: ZVCF LHRH Force Quivers
figure(1);
PL = [ZVCFQ.LHx, ZVCFQ.LHy, ZVCFQ.LHz];
VL = ZVCFQ.LHonClubFGlobal;
PR = [ZVCFQ.RHx, ZVCFQ.RHy, ZVCFQ.RHz];
VR = ZVCFQ.RHonClubFGlobal;
P = [ZVCFQ.MPx, ZVCFQ.MPy, ZVCFQ.MPz];
VT = ZVCFQ.TotalHandForceGlobal;
overlayQuiverForces({PL, PR, P}, {VL, VR, VT}, 'Tags', {'LH Force','RH Force','Net Force'}, 'Colors', {[0 0 1],[1 0 0],[0 1 0]}, 'Title', 'ZVCF LHRH Hand Forces');
savefig('ZVCF Quiver Plots/ZVCF_Quiver Plot - Hand Forces');

%% Plot 2: ZVCF vs DELTA Comparison
figure(2);
VLd = DELTAQ.LHonClubFGlobal;
VRd = DELTAQ.RHonClubFGlobal;
VTd = DELTAQ.TotalHandForceGlobal;

PBL = [BASEQ.LHx, BASEQ.LHy, BASEQ.LHz];
PBR = [BASEQ.RHx, BASEQ.RHy, BASEQ.RHz];
PBM = [BASEQ.MPx, BASEQ.MPy, BASEQ.MPz];

overlayQuiverForces({PL, PR, P, PBL, PBR, PBM}, {VL, VR, VT, VLd, VRd, VTd}, ...
    'Tags', {'ZVCF LH Force','ZVCF RH Force','ZVCF Net Force','DELTA LH Force','DELTA RH Force','DELTA Net Force'}, ...
    'Colors', {[0 0 1],[1 0 0],[0 1 0],[0 0 0.5],[0.5 0 0],[0 0.5 0]}, ...
    'Title', 'ZVCF vs DELTA Hand Forces');
savefig('ZVCF Quiver Plots/ZVCF_Quiver Plot - Hand Forces ZVCF Comparison to Delta');

end
