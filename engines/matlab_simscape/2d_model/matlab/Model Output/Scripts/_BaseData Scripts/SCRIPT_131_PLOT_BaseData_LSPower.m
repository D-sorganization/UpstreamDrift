figure(131);
hold on;

plot(BASEQ.Time,BASEQ.LSonArmLinearPower);
plot(BASEQ.Time,BASEQ.LSonArmAngularPower);
plot(BASEQ.Time,BASEQ.TotalLSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Shoulder Power');
pause(PauseTime);

%Close Figure
close(131);