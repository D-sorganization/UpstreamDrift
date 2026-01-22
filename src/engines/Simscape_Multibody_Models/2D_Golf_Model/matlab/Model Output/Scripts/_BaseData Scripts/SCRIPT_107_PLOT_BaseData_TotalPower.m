figure(107);
hold on;

plot(BASEQ.Time,BASEQ.TotalLSPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);
plot(BASEQ.Time,BASEQ.TotalREPower);
plot(BASEQ.Time,BASEQ.TotalLWPower);
plot(BASEQ.Time,BASEQ.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LW Total Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Total Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Total Power');
pause(PauseTime);

%Close Figure
close(107);