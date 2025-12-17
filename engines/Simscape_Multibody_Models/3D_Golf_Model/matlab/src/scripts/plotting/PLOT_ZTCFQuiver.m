function PLOT_ZTCFQuiver(ZTCFQ)
%PLOT_ZTCFQuiverSuite - Unified plotting script to replicate all ZTCF quiver plots from MASTER_SCRIPT_ZTCFCharts_3D.m
% Uses overlayQuiverForces or shelledQuivers3 for clarity and modularity.

P = [ZTCFQ.MPx, ZTCFQ.MPy, ZTCFQ.MPz];
PL = [ZTCFQ.LHx, ZTCFQ.LHy, ZTCFQ.LHz];
PR = [ZTCFQ.RHx, ZTCFQ.RHy, ZTCFQ.RHz];
F = ZTCFQ.TotalHandForceGlobal;
C = ZTCFQ.EquivalentMidpointCoupleGlobal;
TL = ZTCFQ.LHonClubTGlobal;
TR = ZTCFQ.RHonClubTGlobal;
TNet = ZTCFQ.TotalHandTorqueGlobal;
MOFL = ZTCFQ.LHMOFonClubGlobal;
MOFR = ZTCFQ.RHMOFonClubGlobal;
MOFTotal = ZTCFQ.MPMOFonClubGlobal;

%% Plot 1: Net Force and Equivalent Couple
figure(1);
overlayQuiverForces({P, P}, {F, C}, 'Tags', {'Net Force', 'Equivalent MP Couple'}, 'Colors', {[0 1 0], [0.8 0.2 0]}, 'Title', 'Net Force & Equivalent Couple');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Net Force and Equivalent MP Couple');

%% Plot 2: Net Force Only
figure(2);
overlayQuiverForces({P}, {F}, 'Tags', {'Net Force'}, 'Colors', {[0 1 0]}, 'Title', 'Net Force');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Net Force');

%% Plot 3: Hand Forces
figure(3);
VL = ZTCFQ.LHonClubFGlobal;
VR = ZTCFQ.RHonClubFGlobal;
overlayQuiverForces({PL, PR, P}, {VL, VR, F}, 'Tags', {'LH Force', 'RH Force', 'Net Force'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Forces');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Hand Forces');

%% Plot 4: Hand Torques
figure(4);
overlayQuiverForces({PL, PR, P}, {TL, TR, TNet}, 'Tags', {'LH Torque', 'RH Torque', 'Net Torque'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Torques');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Hand Torques');

%% Plot 5: All Torques and MOFs
figure(5);
overlayQuiverForces({PL, PR, P, PL, PR, P}, {MOFL, MOFR, MOFTotal, TL, TR, TNet}, 'Tags', {'LH MOF','RH MOF','Total MOF','LH Torque','RH Torque','Net Torque'}, 'Colors', {[0 0.5 0],[0.5 0 0],[0 0 0.5],[0 0 1],[1 0 0],[0 1 0]}, 'Title', 'All Torques and Moments');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Torques and Moments');

%% Plot 6: LHRH MOFs Only
figure(6);
overlayQuiverForces({PL, PR}, {MOFL, MOFR}, 'Tags', {'LH MOF','RH MOF'}, 'Colors', {[0 0.5 0],[0.5 0 0]}, 'Title', 'LHRH Moments of Force');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - LHRH Moments of Force');

%% Plot 7: Club COM Dots
figure(7);
COM = ZTCFQ.ClubCOM;
overlayQuiverForces({COM}, {zeros(size(COM))}, 'Tags', {'COM'}, 'Colors', {[0.25 0.25 0.25]}, 'Title', 'Club COM Points');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Club COM Dots');

%% Plot 8: Arm Segments
figure(8);
LA = [ZTCFQ.LSx, ZTCFQ.LSy, ZTCFQ.LSz];
RA = [ZTCFQ.RSx, ZTCFQ.RSy, ZTCFQ.RSz];
LAvec = [ZTCFQ.LeftArmdx, ZTCFQ.LeftArmdy, ZTCFQ.LeftArmdz];
RAvec = [ZTCFQ.RightArmdx, ZTCFQ.RightArmdy, ZTCFQ.RightArmdz];
overlayQuiverForces({LA, RA}, {LAvec, RAvec}, 'Tags', {'Left Arm', 'Right Arm'}, 'Colors', {[0 0 0.5], [0 0.5 0]}, 'Title', 'Upper Arms');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Upper Arms');

%% Plot 9: Forearm Segments
figure(9);
LF = [ZTCFQ.LEx, ZTCFQ.LEy, ZTCFQ.LEz];
RF = [ZTCFQ.REx, ZTCFQ.REy, ZTCFQ.REz];
LFvec = [ZTCFQ.LeftForearmdx, ZTCFQ.LeftForearmdy, ZTCFQ.LeftForearmdz];
RFvec = [ZTCFQ.RightForearmdx, ZTCFQ.RightForearmdy, ZTCFQ.RightForearmdz];
overlayQuiverForces({LF, RF}, {LFvec, RFvec}, 'Tags', {'Left Forearm', 'Right Forearm'}, 'Colors', {[0 0 0.5], [0 0.5 0]}, 'Title', 'Forearms');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Forearms');

%% Plot 10: Total Linear Impulse
figure(10);
LI = ZTCFQ.LinearImpulseonClub;
overlayQuiverForces({P}, {LI}, 'Tags', {'Linear Impulse'}, 'Colors', {[0 1 0]}, 'Title', 'Total Linear Impulse on Club');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Total Linear Impulse on Club');

%% Plot 11: Total Angular Impulse
figure(11);
AI = ZTCFQ.LHAngularImpulseonClub + ZTCFQ.RHAngularImpulseonClub;
overlayQuiverForces({P}, {AI}, 'Tags', {'Angular Impulse'}, 'Colors', {[0 1 0]}, 'Title', 'Total Angular Impulse on Club');
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Total Angular Impulse on Club');

end
