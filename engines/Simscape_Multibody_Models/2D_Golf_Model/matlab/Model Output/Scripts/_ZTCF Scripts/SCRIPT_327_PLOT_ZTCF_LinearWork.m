figure(327);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Work');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Work');
pause(PauseTime);

%Close Figure
close(327);