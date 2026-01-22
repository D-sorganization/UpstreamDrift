function PLOT_DELTAQuiver(BASEQ, DELTAQ, ZTCFQ)
%PLOT_DELTAQuiver - Replicates all quiver plots from MASTER_SCRIPT_DeltaCharts_3D.m

% === Plot 1: Net Force and Equivalent Couple ===
figure(1);
P = [ZTCFQ.MPx, ZTCFQ.MPy, ZTCFQ.MPz];
F = DELTAQ.TotalHandForceGlobal;
C = DELTAQ.EquivalentMidpointCoupleGlobal;
overlayQuiverForces({P, P}, {F, C}, 'Tags', {'Net Force', 'Equivalent MP Couple'}, 'Colors', {[0 1 0], [0.8 0.2 0]}, 'Title', 'Net Force & Equivalent Couple');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Net Force and Equivalent MP Couple');

% === Plot 2: Net Force ===
figure(2);
overlayQuiverForces({P}, {F}, 'Tags', {'Net Force'}, 'Colors', {[0 1 0]}, 'Title', 'Net Force');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Net Force');

% === Plot 3: Hand Forces ===
figure(3);
PL = [ZTCFQ.LHx, ZTCFQ.LHy, ZTCFQ.LHz];
VL = DELTAQ.LHonClubFGlobal;
PR = [ZTCFQ.RHx, ZTCFQ.RHy, ZTCFQ.RHz];
VR = DELTAQ.RHonClubFGlobal;
overlayQuiverForces({PL, PR, P}, {VL, VR, F}, 'Tags', {'LH Force', 'RH Force', 'Net Force'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Forces');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Hand Forces');

% === Plot 4: Hand Torques ===
figure(4);
TL = DELTAQ.LHonClubTGlobal;
TR = DELTAQ.RHonClubTGlobal;
TNet = DELTAQ.TotalHandTorqueGlobal;
overlayQuiverForces({PL, PR, P}, {TL, TR, TNet}, 'Tags', {'LH Torque', 'RH Torque', 'Net Torque'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Torques');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Hand Torques');

% === Plot 5: All Torques and MOFs ===
figure(5);
MOFL = DELTAQ.LHMOFonClubGlobal;
MOFR = DELTAQ.RHMOFonClubGlobal;
MOFTotal = DELTAQ.MPMOFonClubGlobal;
overlayQuiverForces({PL, PR, P, PL, PR, P}, {MOFL, MOFR, MOFTotal, TL, TR, TNet}, 'Tags', {'LH MOF','RH MOF','Total MOF','LH Torque','RH Torque','Net Torque'}, 'Colors', {[0 0.5 0],[0.5 0 0],[0 0 0.5],[0 0 1],[1 0 0],[0 1 0]}, 'Title', 'All Torques and Moments');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Torques and Moments');

% === Plot 6: LHRH MOFs Only ===
figure(6);
overlayQuiverForces({PL, PR}, {MOFL, MOFR}, 'Tags', {'LH MOF','RH MOF'}, 'Colors', {[0 0.5 0],[0.5 0 0]}, 'Title', 'LHRH Moments of Force');
savefig('Delta Quiver Plots/DELTA_Quiver Plot - LHRH Moments of Force');

% Additional limb-specific MOFs and impulse plots will follow this structure...

end
