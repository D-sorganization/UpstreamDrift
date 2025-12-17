figure(123);
hold on;
plot(BASEQ.Time,BASEQ.("CHS (mph)"));
plot(BASEQ.Time,BASEQ.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';

%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');

%Add a Title
title('Clubhead and Hand Speed');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - CHS and Hand Speed');
pause(PauseTime);

%Close Figure
close(123);