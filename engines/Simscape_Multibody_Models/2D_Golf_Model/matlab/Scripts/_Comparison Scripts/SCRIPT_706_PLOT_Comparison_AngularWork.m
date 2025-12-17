figure(706);
hold on;

plot(BASEQ.Time,BASEQ.LSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.RSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.LEAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.REAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.LWAngularWorkonClub);
plot(BASEQ.Time,BASEQ.RWAngularWorkonClub);

plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.LWAngularWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RWAngularWorkonClub,'--');

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Work - BASE','RS Angular Work - BASE','LE Angular Work - BASE','RE Angular Work - BASE','LW Angular Work - BASE','RW Angular Work - BASE','LS Angular Work - ZTCF','RS Angular Work - ZTCF','LE Angular Work - ZTCF','RE Angular Work - ZTCF','LW Angular Work - ZTCF','RW Angular Work - ZTCF');
legend('Location','southeast');

%Add a Title
title('Angular Work on Distal Segment');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Work on Distal');
pause(PauseTime);

%Close Figure
close(706);