figure(11);
hold on;
plot(BASEQ.Time,BASEQ.ForceAlongHandPath);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath);
plot(DELTAQ.Time,DELTAQ.ForceAlongHandPath); 

xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('BASE','ZTCF','DELTA');
legend('Location','southeast');

%Add a Title
title('Force Along Hand Path');
subtitle('Comparison of BASE, ZTCF, DELTA');

%Save Figure
savefig('Comparison Charts/COMPARISON_Plot - Force Along Hand Path');

%Close Figure
close(11);