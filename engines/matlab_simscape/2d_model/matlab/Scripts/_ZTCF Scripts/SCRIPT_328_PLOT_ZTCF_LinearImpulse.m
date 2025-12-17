figure(328);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LHLinearImpulseonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearImpulseonClub);
plot(ZTCFQ.Time,ZTCFQ.LinearImpulseonClub);

ylabel('Linear Impulse (kgm/s)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);

%Close Figure
close(328);