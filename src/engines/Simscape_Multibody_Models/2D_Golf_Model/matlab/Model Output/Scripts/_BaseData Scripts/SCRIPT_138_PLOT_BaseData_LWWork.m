figure(138);
hold on;

plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.LWAngularWorkonClub);
plot(BASEQ.Time,BASEQ.TotalLWWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Work','LW Angular Work','LW Total Work');
legend('Location','southeast');

%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Wrist Work');
pause(PauseTime);

%Close Figure
close(138);