figure(529);
hold on;

EQMPLOCAL=DELTAQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=DELTAQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=DELTAQ.SumofMomentsonClubLocal(:,3);

plot(ZTCFQ.Time,EQMPLOCAL);
plot(ZTCFQ.Time,MPMOFLOCAL);
plot(ZTCFQ.Time,SUMOFMOMENTSLOCAL);

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
subtitle('DELTA - Grip Reference Frame');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Equivalent Couple and MOF');
pause(PauseTime);

%Close Figure
close(529);