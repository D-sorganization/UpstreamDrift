figure(937);
hold on;

plot(Data.Time,Data.REonForearmLinearPower);
plot(Data.Time,Data.REonForearmAngularPower);
plot(Data.Time,Data.TotalREPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');

%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Right Elbow Power');
pause(PauseTime);

%Close Figure
close(937);