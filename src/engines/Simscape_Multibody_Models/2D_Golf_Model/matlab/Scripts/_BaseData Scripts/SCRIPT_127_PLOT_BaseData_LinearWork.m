figure(127);
hold on;

plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.LinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Work');
subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Work on Club');
subtitle('BaseData');
pause(PauseTime);

%Close Figure
close(127);