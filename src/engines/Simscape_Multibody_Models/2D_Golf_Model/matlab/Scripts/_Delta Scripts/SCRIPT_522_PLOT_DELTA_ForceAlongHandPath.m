figure(522);
plot(ZTCFQ.Time,DELTAQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Force Along Hand Path');
pause(PauseTime);

%Close Figure
close(522);