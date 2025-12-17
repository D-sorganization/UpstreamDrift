function readJointStateTargets_GolfSwing3D()
% READJOINTSTATETARGETS_GOLFSWING3D Reads and displays joint priority targets.
%   READJOINTSTATETARGETS_GOLFSWING3D() scans the 'GolfSwing3D_Kinetic' model
%   and prints the priority settings for Revolute, Universal, and Gimbal joints.

    arguments
    end

    disp('🔍 Reading joint priority values in GolfSwing3D_Kinetic...');

    %% Revolute Joints
    revoluteJoints = {
        'GolfSwing3D_Kinetic/Left Elbow Joint/Revolute Joint';
        'GolfSwing3D_Kinetic/Right Elbow Joint/Revolute Joint';
        'GolfSwing3D_Kinetic/Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint';
        'GolfSwing3D_Kinetic/Left Forearm/Revolute Joint';
        'GolfSwing3D_Kinetic/Right Forearm/Revolute Joint';
    };

    for i = 1:length(revoluteJoints)
        joint = revoluteJoints{i};
        pos = get_param(joint, 'RzPositionTargetPriority');
        vel = get_param(joint, 'RzVelocityTargetPriority');
        fprintf('[Revolute] %s\n  Rz Pos Priority: %s | Rz Vel Priority: %s\n', joint, pos, vel);
    end

    %% Universal Joints
    universalJoints = {
        'GolfSwing3D_Kinetic/Left Scapula Joint/Universal Joint';
        'GolfSwing3D_Kinetic/Right Scapula Joint/Universal Joint';
        'GolfSwing3D_Kinetic/Left Wrist and Hand/Universal Joint';
        'GolfSwing3D_Kinetic/Right Wrist and Hand/Universal Joint';
        'GolfSwing3D_Kinetic/Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint';
    };

    for i = 1:length(universalJoints)
        joint = universalJoints{i};
        rxPos = get_param(joint, 'RxPositionTargetPriority');
        ryPos = get_param(joint, 'RyPositionTargetPriority');
        rxVel = get_param(joint, 'RxVelocityTargetPriority');
        ryVel = get_param(joint, 'RyVelocityTargetPriority');
        fprintf('[Universal] %s\n  Rx Pos: %s Vel: %s | Ry Pos: %s Vel: %s\n', ...
            joint, rxPos, rxVel, ryPos, ryVel);
    end

    %% Gimbal Joints
    gimbalJoints = {
        'GolfSwing3D_Kinetic/Left Shoulder Joint/Gimbal Joint';
        'GolfSwing3D_Kinetic/Right Shoulder Joint/Gimbal Joint';
    };

    for i = 1:length(gimbalJoints)
        joint = gimbalJoints{i};
        rxPos = get_param(joint, 'RxPositionTargetPriority');
        ryPos = get_param(joint, 'RyPositionTargetPriority');
        rzPos = get_param(joint, 'RzPositionTargetPriority');
        rxVel = get_param(joint, 'RxVelocityTargetPriority');
        ryVel = get_param(joint, 'RyVelocityTargetPriority');
        rzVel = get_param(joint, 'RzVelocityTargetPriority');
        fprintf('[Gimbal] %s\n  Rx Pos: %s Vel: %s | Ry Pos: %s Vel: %s | Rz Pos: %s Vel: %s\n', ...
            joint, rxPos, rxVel, ryPos, ryVel, rzPos, rzVel);
    end

    disp('✅ Done reading all joint priorities.');
end
