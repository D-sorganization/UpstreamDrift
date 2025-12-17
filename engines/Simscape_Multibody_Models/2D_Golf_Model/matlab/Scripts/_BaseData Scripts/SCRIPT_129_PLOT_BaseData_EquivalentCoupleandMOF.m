figure(129);
hold on;

EQMPLOCAL=BASEQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=BASEQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=BASEQ.SumofMomentsonClubLocal(:,3);

plot(BASEQ.Time,EQMPLOCAL);
plot(BASEQ.Time,MPMOFLOCAL);
plot(BASEQ.Time,SUMOFMOMENTSLOCAL);

clear("EQMPLOCAL");
clear("MPMOFLOCAL");
clear("SUMOFMOMENTSLOCAL");

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('Equivalent Midpoint Couple','Total Force on Midpoint MOF','Sum of Moments');
legend('Location','southeast');

%Add a Title
title('Equivalent Couple, Moment of Force, Sum of Moments');
subtitle('BASE - Grip Reference Frame');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Equivalent Couple and MOF');
pause(PauseTime);

%Close Figure
close(129);