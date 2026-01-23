figure(712);
hold on;

plot(BASEQ.Time,BASEQ.LWAngularWorkonClub);
plot(BASEQ.Time,BASEQ.RWAngularWorkonClub);


plot(ZTCFQ.Time,ZTCFQ.LWAngularWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RWAngularWorkonClub,'--');

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LW Angular Work - BASE','RW Angular Work - BASE','LW Angular Work - ZTCF','RW Angular Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Angular Work on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Work on Club');
pause(PauseTime);

%Close Figure
close(712);