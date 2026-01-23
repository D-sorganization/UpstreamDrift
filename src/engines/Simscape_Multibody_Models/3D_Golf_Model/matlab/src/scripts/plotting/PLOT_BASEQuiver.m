function PLOT_BASEQuiver(BASEQ)
%PLOT_BASEQuiver - Full master plot for all BASE quiver plots
% Mirrors MASTER_SCRIPT_BaseDataCharts_3D.m using overlayQuiverForces

%% Plot 1: Net Force and Equivalent Couple
figure(1);
P = [BASEQ.MPx, BASEQ.MPy, BASEQ.MPz];
F = BASEQ.TotalHandForceGlobal;
C = BASEQ.EquivalentMidpointCoupleGlobal;
overlayQuiverForces({P, P}, {F, C}, 'Tags', {'Net Force', 'Equivalent MP Couple'}, 'Colors', {[0 1 0], [0.8 0.2 0]}, 'Title', 'Net Force & Equivalent Couple');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Net Force and Equivalent MP Couple');

%% Plot 2: Net Force
figure(2);
overlayQuiverForces({P}, {F}, 'Tags', {'Net Force'}, 'Colors', {[0 1 0]}, 'Title', 'Net Force');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Net Force');

%% Plot 3: Hand Forces
figure(3);
PL = [BASEQ.LHx, BASEQ.LHy, BASEQ.LHz];
VL = BASEQ.LHonClubFGlobal;
PR = [BASEQ.RHx, BASEQ.RHy, BASEQ.RHz];
VR = BASEQ.RHonClubFGlobal;
overlayQuiverForces({PL, PR, P}, {VL, VR, F}, 'Tags', {'LH Force', 'RH Force', 'Net Force'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Forces');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Hand Forces');

%% Plot 4: Hand Torques
figure(4);
TL = BASEQ.LHonClubTGlobal;
TR = BASEQ.RHonClubTGlobal;
TNet = BASEQ.TotalHandTorqueGlobal;
overlayQuiverForces({PL, PR, P}, {TL, TR, TNet}, 'Tags', {'LH Torque', 'RH Torque', 'Net Torque'}, 'Colors', {[0 0 1], [1 0 0], [0 1 0]}, 'Title', 'Total Hand Torques');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Hand Torques');

%% Plot 5: All Torques and MOFs
figure(5);
MOFL = BASEQ.LHMOFonClubGlobal;
MOFR = BASEQ.RHMOFonClubGlobal;
MOFTotal = BASEQ.MPMOFonClubGlobal;
overlayQuiverForces({PL, PR, P, PL, PR, P}, {MOFL, MOFR, MOFTotal, TL, TR, TNet}, 'Tags', {'LH MOF','RH MOF','Total MOF','LH Torque','RH Torque','Net Torque'}, 'Colors', {[0 0.5 0],[0.5 0 0],[0 0 0.5],[0 0 1],[1 0 0],[0 1 0]}, 'Title', 'All Torques and Moments');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Torques and Moments');

%% Plot 6: LHRH MOFs Only
figure(6);
overlayQuiverForces({PL, PR}, {MOFL, MOFR}, 'Tags', {'LH MOF','RH MOF'}, 'Colors', {[0 0.5 0],[0.5 0 0]}, 'Title', 'LHRH Moments of Force');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - LHRH Moments of Force');

%% Plot 7: Club COM Dots
figure(7);
COM = BASEQ.ClubCOM;
overlayQuiverForces({COM}, {zeros(size(COM))}, 'Tags', {'COM'}, 'Colors', {[0.25 0.25 0.25]}, 'Title', 'Club COM Points');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Club COM Dots');

%% Plot 8: Upper Arm Forces and MOFs (Right Only Shown)
figure(8);

PRA = [BASEQ.RSx, BASEQ.RSy, BASEQ.RSz];
ERA = [BASEQ.REx, BASEQ.REy, BASEQ.REz];
VRA = [BASEQ.RightArmdx, BASEQ.RightArmdy, BASEQ.RightArmdz];
F1 = [BASEQ.RForearmonRArmFGlobal1, BASEQ.RForearmonRArmFGlobal2, BASEQ.RForearmonRArmFGlobal3];
F2 = [BASEQ.RSonRArmFGlobal1, BASEQ.RSonRArmFGlobal2, BASEQ.RSonRArmFGlobal3];
M1 = [BASEQ.RElbowonRArmMOFGlobal1, BASEQ.RElbowonRArmMOFGlobal2, BASEQ.RElbowonRArmMOFGlobal3];
M2 = [BASEQ.RShoulderonRArmMOFGlobal1, BASEQ.RShoulderonRArmMOFGlobal2, BASEQ.RShoulderonRArmMOFGlobal3];
overlayQuiverForces({ERA, PRA, ERA, PRA, PRA}, {F1, F2, M1, M2, VRA}, 'Tags', {'RE Force','RS Force','RE MOF','RS MOF','Arm Vector'}, 'Colors', {[0 0 1],[1 0 0],[0 0.75 0],[0 0.5 0],[0 0 0]}, 'Title', 'Right Arm Forces and MOFs');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Right Upper Arm Moments');

%% SCRIPT_118-121 Moment Snapshots
% Right Upper Arm MOF
figure(11);
overlayQuiverForces({[BASEQ.RSx, BASEQ.RSy, BASEQ.RSz]}, {[BASEQ.RShoulderonRArmMOFGlobal1, BASEQ.RShoulderonRArmMOFGlobal2, BASEQ.RShoulderonRArmMOFGlobal3]}, 'Tags', {'RS MOF'}, 'Colors', {[0 0.5 0]}, 'Title', 'Right Upper Arm Moment');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Right Upper Arm MOF');

% Left Upper Arm MOF
figure(12);
overlayQuiverForces({[BASEQ.LSx, BASEQ.LSy, BASEQ.LSz]}, {[BASEQ.LShoulderonLArmMOFGlobal1, BASEQ.LShoulderonLArmMOFGlobal2, BASEQ.LShoulderonLArmMOFGlobal3]}, 'Tags', {'LS MOF'}, 'Colors', {[0.25 0.25 0.8]}, 'Title', 'Left Upper Arm Moment');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Left Upper Arm MOF');

% Right Forearm MOF
figure(13);
overlayQuiverForces({[BASEQ.REx, BASEQ.REy, BASEQ.REz]}, {[BASEQ.RElbowonRArmMOFGlobal1, BASEQ.RElbowonRArmMOFGlobal2, BASEQ.RElbowonRArmMOFGlobal3]}, 'Tags', {'RE MOF'}, 'Colors', {[0.5 0.2 0.6]}, 'Title', 'Right Forearm Moment');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Right Forearm MOF');

% Left Forearm MOF
figure(14);
overlayQuiverForces({[BASEQ.LEx, BASEQ.LEy, BASEQ.LEz]}, {[BASEQ.LElbowonLArmMOFGlobal1, BASEQ.LElbowonLArmMOFGlobal2, BASEQ.LElbowonLArmMOFGlobal3]}, 'Tags', {'LE MOF'}, 'Colors', {[0.4 0.4 1]}, 'Title', 'Left Forearm Moment');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Left Forearm MOF');

%% Plot 9: Linear Impulse
figure(9);
LI = BASEQ.LinearImpulseonClub;
overlayQuiverForces({P}, {LI}, 'Tags', {'Linear Impulse'}, 'Colors', {[0 1 0]}, 'Title', 'Total Linear Impulse on Club');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Total Linear Impulse on Club');

%% Plot 10: Angular Impulse
figure(10);
AI = BASEQ.LHAngularImpulseonClub + BASEQ.RHAngularImpulseonClub;
overlayQuiverForces({P}, {AI}, 'Tags', {'Angular Impulse'}, 'Colors', {[0 1 0]}, 'Title', 'Total Angular Impulse on Club');
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Total Angular Impulse on Club');

end
