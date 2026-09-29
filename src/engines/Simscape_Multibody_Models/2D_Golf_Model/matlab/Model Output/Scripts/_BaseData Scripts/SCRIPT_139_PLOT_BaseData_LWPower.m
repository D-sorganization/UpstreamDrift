figure(139);
hold on;

plot(BASEQ.Time,BASEQ.LWonClubLinearPower);
plot(BASEQ.Time,BASEQ.LWonClubAngularPower);
plot(BASEQ.Time,BASEQ.TotalLWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Power','LW Angular Power','LW Total Power');
legend('Location','southeast');

%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Wrist Power');
pause(PauseTime);

%Close Figure
close(139);