figure(540);
hold on;

plot(DELTAQ.Time,DELTAQ.RHLinearWorkonClub);
plot(DELTAQ.Time,DELTAQ.RWAngularWorkonClub);
plot(DELTAQ.Time,DELTAQ.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Work','RW Angular Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Wrist Work');
pause(PauseTime);

%Close Figure
close(540);