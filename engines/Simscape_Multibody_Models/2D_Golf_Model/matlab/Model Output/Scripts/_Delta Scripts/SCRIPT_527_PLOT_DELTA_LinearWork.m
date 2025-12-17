figure(527);
hold on;

plot(ZTCFQ.Time,DELTAQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.LinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Work');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Work');
pause(PauseTime);

%Close Figure
close(527);