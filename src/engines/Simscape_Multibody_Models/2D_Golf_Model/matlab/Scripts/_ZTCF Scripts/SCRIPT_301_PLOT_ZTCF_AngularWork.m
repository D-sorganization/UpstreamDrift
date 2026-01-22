figure(301);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LWAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RWAngularWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LW Angular Work','RW Angular Work');
legend('Location','southeast');

%Add a Title
title('Angular Work on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Angular Work');
pause(PauseTime);

%Close Figure
close(301);