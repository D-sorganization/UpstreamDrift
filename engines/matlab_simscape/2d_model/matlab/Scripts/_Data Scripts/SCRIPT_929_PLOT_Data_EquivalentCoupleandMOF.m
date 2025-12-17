figure(929);
hold on;

EQMPLOCAL=Data.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=Data.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=Data.SumofMomentsonClubLocal(:,3);

plot(Data.Time,EQMPLOCAL);
plot(Data.Time,MPMOFLOCAL);
plot(Data.Time,SUMOFMOMENTSLOCAL);

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
subtitle('Data - Grip Reference Frame');

%Save Figure
savefig('Data Charts/Plot - Equivalent Couple and MOF');
pause(PauseTime);

%Close Figure
close(929);