figure(707);
hold on;

plot(BASEQ.Time,BASEQ.TotalLSWork);
plot(BASEQ.Time,BASEQ.TotalRSWork);
plot(BASEQ.Time,BASEQ.TotalLEWork);
plot(BASEQ.Time,BASEQ.TotalREWork);
plot(BASEQ.Time,BASEQ.TotalLWWork);
plot(BASEQ.Time,BASEQ.TotalRWWork);

plot(ZTCFQ.Time,ZTCFQ.TotalLSWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalREWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLWWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRWWork,'--');

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Total Work - BASE','RS Total Work - BASE','LE Total Work - BASE','RE Total Work - BASE','LW Total Work - BASE','RW Total Work - BASE','LS Total Work - ZTCF','RS Total Work - ZTCF','LE Total Work - ZTCF','RE Total Work - ZTCF','LW Total Work - ZTCF','RW Total Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Total Work on Distal Segment');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Work on Distal');
pause(PauseTime);

%Close Figure
close(707);