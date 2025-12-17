figure(302);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.RSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.REonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LWonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.RWonClubAngularPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LW Angular Power','RW Angular Power');
legend('Location','southeast');

%Add a Title
title('Angular Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Angular Power');
pause(PauseTime);

%Close Figure
close(302);