figure(340);
hold on;

plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RWAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Work','RW Angular Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Wrist Work');
pause(PauseTime);

%Close Figure
close(340);