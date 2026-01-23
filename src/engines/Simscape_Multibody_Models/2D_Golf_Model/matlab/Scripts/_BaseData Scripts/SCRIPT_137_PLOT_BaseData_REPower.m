figure(137);
hold on;

plot(BASEQ.Time,BASEQ.REonForearmLinearPower);
plot(BASEQ.Time,BASEQ.REonForearmAngularPower);
plot(BASEQ.Time,BASEQ.TotalREPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');

%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Elbow Power');
pause(PauseTime);

%Close Figure
close(137);