%Reads from a Table named ZTCF in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate LeftForearm Quivers
LeftForearm=quiver3(ZTCFQ.LEx(:,1),ZTCFQ.LEy(:,1),ZTCFQ.LEz(:,1),ZTCFQ.LeftForearmdx(:,1),ZTCFQ.LeftForearmdy(:,1),ZTCFQ.LeftForearmdz(:,1),0);
LeftForearm.ShowArrowHead='off';		
LeftForearm.LineWidth=1;			   
LeftForearm.Color=[0 0 0.5];		
hold on;				        

%Generate Right Forearm Quivers
RightForearm=quiver3(ZTCFQ.REx(:,1),ZTCFQ.REy(:,1),ZTCFQ.REz(:,1),ZTCFQ.RightForearmdx(:,1),ZTCFQ.RightForearmdy(:,1),ZTCFQ.RightForearmdz(:,1),0);
RightForearm.ShowArrowHead='off';
RightForearm.LineWidth=1;			   
RightForearm.Color=[0 0.5 0];			   

%Generate Left Arm Quivers
LeftArm=quiver3(ZTCFQ.LSx(:,1),ZTCFQ.LSy(:,1),ZTCFQ.LSz(:,1),ZTCFQ.LeftArmdx(:,1),ZTCFQ.LeftArmdy(:,1),ZTCFQ.LeftArmdz(:,1),0);
LeftArm.ShowArrowHead='off';		
LeftArm.LineWidth=1;			   
LeftArm.Color=[0 0 0.5];		

%Generate Right Arm Quivers
RightArm=quiver3(ZTCFQ.RSx(:,1),ZTCFQ.RSy(:,1),ZTCFQ.RSz(:,1),ZTCFQ.RightArmdx(:,1),ZTCFQ.RightArmdy(:,1),ZTCFQ.RightArmdz(:,1),0);
RightArm.ShowArrowHead='off';
RightArm.LineWidth=1;			   
RightArm.Color=[0 0.5 0];	

%Generate Left Shoulder Quivers
LeftShoulder=quiver3(ZTCFQ.HUBx(:,1),ZTCFQ.HUBy(:,1),ZTCFQ.HUBz(:,1),ZTCFQ.LeftShoulderdx(:,1),ZTCFQ.LeftShoulderdy(:,1),ZTCFQ.LeftShoulderdz(:,1),0);
LeftShoulder.ShowArrowHead='off';		
LeftShoulder.LineWidth=1;			   
LeftShoulder.Color=[0 0 0.5];		

%Generate Right Shoulder Quivers
RightShoulder=quiver3(ZTCFQ.HUBx(:,1),ZTCFQ.HUBy(:,1),ZTCFQ.HUBz(:,1),ZTCFQ.RightShoulderdx(:,1),ZTCFQ.RightShoulderdy(:,1),ZTCFQ.RightShoulderdz(:,1),0);
RightShoulder.ShowArrowHead='off';
RightShoulder.LineWidth=1;			   
RightShoulder.Color=[0 0.5 0];	