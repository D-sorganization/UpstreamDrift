figure(140);
hold on;

plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RWAngularWorkonClub);
plot(BASEQ.Time,BASEQ.TotalRWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Work','RW Angular Work','RW Total Work');
legend('Location','southeast');

%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Wrist Work');
pause(PauseTime);

%Close Figure
close(140);