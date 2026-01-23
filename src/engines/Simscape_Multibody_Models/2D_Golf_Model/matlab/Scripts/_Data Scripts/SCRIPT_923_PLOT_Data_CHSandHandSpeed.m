figure(923);
hold on;
plot(Data.Time,Data.("CHS (mph)"));
plot(Data.Time,Data.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';

%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');

%Add a Title
title('Clubhead and Hand Speed');
subtitle('Data');

%Save Figure
savefig('Data Charts/Plot - CHS and Hand Speed');
pause(PauseTime);

%Close Figure
close(923);