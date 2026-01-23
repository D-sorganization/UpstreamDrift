figure(704);
hold on;
plot(BASEQ.Time,BASEQ.ForceAlongHandPath,'LineWidth',3);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath,'--','LineWidth',3);
plot(DELTAQ.Time,DELTAQ.ForceAlongHandPath,':','LineWidth',3); 

xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('BASE','ZTCF','DELTA');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_Plot - Force Along Hand Path');
pause(PauseTime);

%Close Figure
close(704);