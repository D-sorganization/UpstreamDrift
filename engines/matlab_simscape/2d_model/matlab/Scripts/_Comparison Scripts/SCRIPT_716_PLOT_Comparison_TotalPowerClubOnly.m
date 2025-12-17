figure(716);
hold on;

plot(BASEQ.Time,BASEQ.TotalLWPower);
plot(BASEQ.Time,BASEQ.TotalRWPower);

plot(ZTCFQ.Time,ZTCFQ.TotalLWPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRWPower,'--');


ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Total Power - BASE','RW Total Power - BASE','LW Total Power - ZTCF','RW Total Power - ZTCF');
legend('Location','southeast');

%Add a Title
title('Total Power on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Power on Club');
pause(PauseTime);

%Close Figure
close(716);