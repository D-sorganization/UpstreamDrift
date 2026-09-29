figure(939);
hold on;

plot(Data.Time,Data.LWonClubLinearPower);
plot(Data.Time,Data.LWonClubAngularPower);
plot(Data.Time,Data.TotalLWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Power','LW Angular Power','LW Total Power');
legend('Location','southeast');

%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Left Wrist Power');
pause(PauseTime);

%Close Figure
close(939);