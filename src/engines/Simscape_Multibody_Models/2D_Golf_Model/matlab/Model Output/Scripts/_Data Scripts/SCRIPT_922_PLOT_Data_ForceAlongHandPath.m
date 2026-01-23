figure(922);
plot(Data.Time,Data.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('Data');

%Save Figure
savefig('Data Charts/Plot - Force Along Hand Path');
pause(PauseTime);

%Close Figure
close(922);