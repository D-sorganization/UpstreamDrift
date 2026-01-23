figure(528);
hold on;

plot(ZTCFQ.Time,DELTAQ.LHLinearImpulseonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearImpulseonClub);
plot(ZTCFQ.Time,DELTAQ.LinearImpulseonClub);

ylabel('Linear Impulse (kgm/s)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);

%Close Figure
close(528);