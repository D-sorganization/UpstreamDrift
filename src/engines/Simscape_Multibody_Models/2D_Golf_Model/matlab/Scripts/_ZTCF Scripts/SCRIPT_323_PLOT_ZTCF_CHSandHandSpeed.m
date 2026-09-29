figure(323);
hold on;
plot(ZTCFQ.Time,ZTCFQ.("CHS (mph)"));
plot(ZTCFQ.Time,ZTCFQ.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';

%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');

%Add a Title
title('Clubhead and Hand Speed');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - CHS and Hand Speed');
pause(PauseTime);

%Close Figure
close(323);