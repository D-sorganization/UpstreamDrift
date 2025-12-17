function setJointStateTargets_GolfSwing3D()
    %% Revolute Joints (1 DoF: Rz)
    revoluteJoints = {
        'GolfSwing3D_Kinetic/Left Elbow Joint/Revolute Joint',        'High', 'Low';
        'GolfSwing3D_Kinetic/Right Elbow Joint/Revolute Joint',       'Low',  'High';
        'GolfSwing3D_Kinetic/Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint', 'High', 'None';
        'GolfSwing3D_Kinetic/Left Forearm/Revolute Joint',            'Low',  'None';
        'GolfSwing3D_Kinetic/Right Forearm/Revolute Joint',           'None', 'Low';
    };

    for i = 1:size(revoluteJoints, 1)
        jointPath = revoluteJoints{i, 1};
        posPriority  = revoluteJoints{i, 2};
        velPriority  = revoluteJoints{i, 3};
        try
            set_param(jointPath, 'RzPositionTargetPriority', posPriority);
            set_param(jointPath, 'RzVelocityTargetPriority', velPriority);
            fprintf('Revolute Joint: %s -> Rz Pos:%s, Vel:%s\n', jointPath, posPriority, velPriority);
        catch ME
            warning('Revolute Joint %s: %s', jointPath, ME.message);
        end
    end

    %% Universal Joints (2 DoF: Rx and Ry)
    universalJoints = {
        'GolfSwing3D_Kinetic/Left Scapula Joint/Universal Joint',     'High', 'Low',  'Low',  'High';
        'GolfSwing3D_Kinetic/Right Scapula Joint/Universal Joint',    'Low',  'High', 'High', 'Low';
        'GolfSwing3D_Kinetic/Left Wrist and Hand/Universal Joint',    'None', 'Low',  'None', 'None';
        'GolfSwing3D_Kinetic/Right Wrist and Hand/Universal Joint',   'Low',  'None', 'Low',  'None';
        'GolfSwing3D_Kinetic/Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint', ...
                                                                       'High', 'High', 'Low',  'High';
    };

    for i = 1:size(universalJoints, 1)
        jointPath = universalJoints{i, 1};
        rxPos = universalJoints{i, 2}; ryPos = universalJoints{i, 3};
        rxVel = universalJoints{i, 4}; ryVel = universalJoints{i, 5};
        try
            set_param(jointPath, 'RxPositionTargetPriority', rxPos);
            set_param(jointPath, 'RyPositionTargetPriority', ryPos);
            set_param(jointPath, 'RxVelocityTargetPriority', rxVel);
            set_param(jointPath, 'RyVelocityTargetPriority', ryVel);
            fprintf('Universal Joint: %s -> Rx Pos:%s, Vel:%s | Ry Pos:%s, Vel:%s\n', ...
                    jointPath, rxPos, rxVel, ryPos, ryVel);
        catch ME
            warning('Universal Joint %s: %s', jointPath, ME.message);
        end
    end

    %% Gimbal Joints (3 DoF: Rx, Ry, Rz)
    gimbalJoints = {
        'GolfSwing3D_Kinetic/Left Shoulder Joint/Gimbal Joint',  'High', 'Low',  'None', 'Low',  'None', 'High';
        'GolfSwing3D_Kinetic/Right Shoulder Joint/Gimbal Joint', 'Low',  'High', 'High', 'None', 'High', 'Low';
    };

    for i = 1:size(gimbalJoints, 1)
        jointPath = gimbalJoints{i, 1};
        rxPos = gimbalJoints{i, 2}; ryPos = gimbalJoints{i, 3}; rzPos = gimbalJoints{i, 4};
        rxVel = gimbalJoints{i, 5}; ryVel = gimbalJoints{i, 6}; rzVel = gimbalJoints{i, 7};
        try
            set_param(jointPath, 'RxPositionTargetPriority', rxPos);
            set_param(jointPath, 'RyPositionTargetPriority', ryPos);
            set_param(jointPath, 'RzPositionTargetPriority', rzPos);
            set_param(jointPath, 'RxVelocityTargetPriority', rxVel);
            set_param(jointPath, 'RyVelocityTargetPriority', ryVel);
            set_param(jointPath, 'RzVelocityTargetPriority', rzVel);
            fprintf('Gimbal Joint: %s -> Rx Pos:%s Vel:%s | Ry Pos:%s Vel:%s | Rz Pos:%s Vel:%s\n', ...
                    jointPath, rxPos, rxVel, ryPos, ryVel, rzPos, rzVel);
        catch ME
            warning('Gimbal Joint %s: %s', jointPath, ME.message);
        end
    end
end
