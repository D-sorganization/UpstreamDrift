figure(306);
hold on;

plot(ZTCFQ.Time,ZTCFQ.TotalLSWork);
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork);
plot(ZTCFQ.Time,ZTCFQ.TotalREWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLWWork);
plot(ZTCFQ.Time,ZTCFQ.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LW Total Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Total Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Total Work');
pause(PauseTime);

%Close Figure
close(306);