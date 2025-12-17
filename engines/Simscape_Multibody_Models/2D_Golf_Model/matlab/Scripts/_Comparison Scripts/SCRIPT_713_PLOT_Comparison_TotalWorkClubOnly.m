figure(713);
hold on;

plot(BASEQ.Time,BASEQ.TotalLWWork);
plot(BASEQ.Time,BASEQ.TotalRWWork);

plot(ZTCFQ.Time,ZTCFQ.TotalLWWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRWWork,'--');

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Total Work - BASE','RW Total Work - BASE','LW Total Work - ZTCF','RW Total Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Total Work Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Work on Club');
pause(PauseTime);

%Close Figure
close(713);