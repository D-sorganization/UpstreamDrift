function short_name = getShortenedJointName(joint_name)
    % External function for creating shortened joint names - can be used in parallel processing
    % This function doesn't rely on handles

    % Create shortened joint names for display
    short_name = strrep(joint_name, 'TorqueInput', 'T');
    short_name = strrep(short_name, 'Input', '');
end
