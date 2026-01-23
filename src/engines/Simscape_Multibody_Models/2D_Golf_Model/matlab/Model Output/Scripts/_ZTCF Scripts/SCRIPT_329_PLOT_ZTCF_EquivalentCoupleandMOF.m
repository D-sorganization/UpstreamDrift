figure(329);
hold on;

EQMPLOCAL=ZTCFQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=ZTCFQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=ZTCFQ.SumofMomentsonClubLocal(:,3);

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
subtitle('ZTCF - Grip Reference Frame');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Equivalent Couple and MOF');
pause(PauseTime);

%Close Figure
close(329);