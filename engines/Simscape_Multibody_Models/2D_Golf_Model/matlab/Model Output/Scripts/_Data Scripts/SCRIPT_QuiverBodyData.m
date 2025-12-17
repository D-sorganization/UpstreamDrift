%Reads from a Table named Data in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate LeftForearm Quivers
LeftForearm=quiver3(Data.LEx(:,1),Data.LEy(:,1),Data.LEz(:,1),Data.LeftForearmdx(:,1),Data.LeftForearmdy(:,1),Data.LeftForearmdz(:,1),0);
LeftForearm.ShowArrowHead='off';		
LeftForearm.LineWidth=1;			   
LeftForearm.Color=[0 0 0.5];		
hold on;				        

%Generate Right Forearm Quivers
RightForearm=quiver3(Data.REx(:,1),Data.REy(:,1),Data.REz(:,1),Data.RightForearmdx(:,1),Data.RightForearmdy(:,1),Data.RightForearmdz(:,1),0);
RightForearm.ShowArrowHead='off';
RightForearm.LineWidth=1;			   
RightForearm.Color=[0 0.5 0];			   

%Generate Left Arm Quivers
LeftArm=quiver3(Data.LSx(:,1),Data.LSy(:,1),Data.LSz(:,1),Data.LeftArmdx(:,1),Data.LeftArmdy(:,1),Data.LeftArmdz(:,1),0);
LeftArm.ShowArrowHead='off';		
LeftArm.LineWidth=1;			   
LeftArm.Color=[0 0 0.5];		

%Generate Right Arm Quivers
RightArm=quiver3(Data.RSx(:,1),Data.RSy(:,1),Data.RSz(:,1),Data.RightArmdx(:,1),Data.RightArmdy(:,1),Data.RightArmdz(:,1),0);
RightArm.ShowArrowHead='off';
RightArm.LineWidth=1;			   
RightArm.Color=[0 0.5 0];	

%Generate Left Shoulder Quivers
LeftShoulder=quiver3(Data.HUBx(:,1),Data.HUBy(:,1),Data.HUBz(:,1),Data.LeftShoulderdx(:,1),Data.LeftShoulderdy(:,1),Data.LeftShoulderdz(:,1),0);
LeftShoulder.ShowArrowHead='off';		
LeftShoulder.LineWidth=1;			   
LeftShoulder.Color=[0 0 0.5];		

%Generate Right Shoulder Quivers
RightShoulder=quiver3(Data.HUBx(:,1),Data.HUBy(:,1),Data.HUBz(:,1),Data.RightShoulderdx(:,1),Data.RightShoulderdy(:,1),Data.RightShoulderdz(:,1),0);
RightShoulder.ShowArrowHead='off';
RightShoulder.LineWidth=1;			   
RightShoulder.Color=[0 0.5 0];	