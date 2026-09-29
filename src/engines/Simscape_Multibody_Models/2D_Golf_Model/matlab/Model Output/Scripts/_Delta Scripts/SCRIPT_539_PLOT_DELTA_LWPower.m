figure(539);
hold on;

plot(DELTAQ.Time,DELTAQ.LWonClubLinearPower);
plot(DELTAQ.Time,DELTAQ.LWonClubAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Power','LW Angular Power','LW Total Power');
legend('Location','southeast');

%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Wrist Power');
pause(PauseTime);

%Close Figure
close(539);