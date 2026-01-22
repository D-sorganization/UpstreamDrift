figure(931);
hold on;

plot(Data.Time,Data.LSonArmLinearPower);
plot(Data.Time,Data.LSonArmAngularPower);
plot(Data.Time,Data.TotalLSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Left Shoulder Power');
pause(PauseTime);

%Close Figure
close(931);