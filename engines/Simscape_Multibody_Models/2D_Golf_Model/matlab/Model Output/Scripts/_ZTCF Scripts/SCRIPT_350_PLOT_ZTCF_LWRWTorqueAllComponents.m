figure(350);
hold on;


LWTORQUEXBASE=BASEQ.LeftWristTorqueLocal(:,1);
LWTORQUEYBASE=BASEQ.LeftWristTorqueLocal(:,2);
LWTORQUEZBASE=BASEQ.LeftWristTorqueLocal(:,3);
LWTORQUEXZTCF=ZTCFQ.LeftWristTorqueLocal(:,1);
LWTORQUEYZTCF=ZTCFQ.LeftWristTorqueLocal(:,2);
LWTORQUEZZTCF=ZTCFQ.LeftWristTorqueLocal(:,3);
LWTORQUEXDELTA=DELTAQ.LeftWristTorqueLocal(:,1);
LWTORQUEYDELTA=DELTAQ.LeftWristTorqueLocal(:,2);
LWTORQUEZDELTA=DELTAQ.LeftWristTorqueLocal(:,3);
RWTORQUEXBASE=BASEQ.RightWristTorqueLocal(:,1);
RWTORQUEYBASE=BASEQ.RightWristTorqueLocal(:,2);
RWTORQUEZBASE=BASEQ.RightWristTorqueLocal(:,3);
RWTORQUEXZTCF=ZTCFQ.RightWristTorqueLocal(:,1);
RWTORQUEYZTCF=ZTCFQ.RightWristTorqueLocal(:,2);
RWTORQUEZZTCF=ZTCFQ.RightWristTorqueLocal(:,3);
RWTORQUEXDELTA=DELTAQ.RightWristTorqueLocal(:,1);
RWTORQUEYDELTA=DELTAQ.RightWristTorqueLocal(:,2);
RWTORQUEZDELTA=DELTAQ.RightWristTorqueLocal(:,3);

plot(ZTCFQ.Time,LWTORQUEXBASE);
plot(ZTCFQ.Time,LWTORQUEYBASE);
plot(ZTCFQ.Time,LWTORQUEZBASE);
plot(ZTCFQ.Time,LWTORQUEXZTCF);
plot(ZTCFQ.Time,LWTORQUEYZTCF);
plot(ZTCFQ.Time,LWTORQUEZZTCF);
plot(DELTAQ.Time,LWTORQUEXDELTA);
plot(DELTAQ.Time,LWTORQUEYDELTA);
plot(DELTAQ.Time,LWTORQUEZDELTA);


plot(ZTCFQ.Time,RWTORQUEXBASE);
plot(ZTCFQ.Time,RWTORQUEYBASE);
plot(ZTCFQ.Time,RWTORQUEZBASE);
plot(ZTCFQ.Time,RWTORQUEXZTCF);
plot(ZTCFQ.Time,RWTORQUEYZTCF);
plot(ZTCFQ.Time,RWTORQUEZZTCF);
plot(DELTAQ.Time,RWTORQUEXDELTA);
plot(DELTAQ.Time,RWTORQUEYDELTA);
plot(DELTAQ.Time,RWTORQUEZDELTA);

clear LWTORQUEXBASE;
clear LWTORQUEYBASE;
clear LWTORQUEZBASE;
clear LWTORQUEXZTCF;
clear LWTORQUEYZTCF;
clear LWTORQUEZZTCF;
clear LWTORQUEXDELTA;
clear LWTORQUEYDELTA;
clear LWTORQUEZDELTA;
clear RWTORQUEXBASE;
clear RWTORQUEYBASE;
clear RWTORQUEZBASE;
clear RWTORQUEXZTCF;
clear RWTORQUEYZTCF;
clear RWTORQUEZZTCF;
clear RWTORQUEXDELTA;
clear RWTORQUEYDELTA;
clear RWTORQUEZDELTA;

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('LWTorqueXBase','LWTorqueYBase','LWTorqueZBase','LWTorqueXZTCF',...
    'LWTorqueYZTCF','LWTorqueZZTCF','LWTorqueXDelta','LWTorqueYDelta','LWTorqueZDelta','RWTorqueXBase','RWTorqueYBase',...
    'RWTorqueZBase','RWTorqueXZTCF','RWTorqueYZTCF','RWTorqueZZTCF''RWTorqueXDelta','RWTorqueYDelta','RWTorqueZDelta');

legend('Location','southeast');

%Add a Title
title('LW and RW Torques All Components');
subtitle('ZTCF - Grip Reference Frame');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - LW RW Torques All Components');
pause(PauseTime);

%Close Figure
% close(350);