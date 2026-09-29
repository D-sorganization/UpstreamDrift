function [points,indices] = select_capture_marker_frame(capture,index,labels)
%SELECT_CAPTURE_MARKER_FRAME Select valid observations in requested label order.
% Invalid payload zeros are absent observations; valid origins remain valid.
% Indices refer to requested labels, for selecting matching body attachments.
    arguments
        capture (1,1) struct
        index (1,1) double {mustBeInteger,mustBePositive}
        labels (:,1) string
    end
    assert(index<=numel(capture.time_s),'select_capture_marker_frame:range', ...
        'Frame index exceeds the capture clock.');
    [found,columns]=ismember(labels,string(capture.labels));
    assert(all(found) && ~isempty(labels) && numel(unique(labels))==numel(labels), ...
        'select_capture_marker_frame:labels','Requested labels must be present and unique.');
    frame_count=numel(capture.time_s);marker_count=numel(capture.labels);
    assert(isequal(size(capture.points_world_m),[frame_count,marker_count,3]) && ...
        isequal(size(capture.valid),[frame_count,marker_count]) && ...
        all(ismember(capture.valid(:),[0,1])), ...
        'select_capture_marker_frame:shape','Capture points and binary validity must share clock and labels.');
    valid=logical(capture.valid(index,columns));indices=find(valid(:));
    assert(~isempty(indices),'select_capture_marker_frame:noObservations', ...
        'No selected marker is valid in this frame.');
    points=reshape(capture.points_world_m(index,columns(indices),:),numel(indices),3);
    assert(isreal(points) && all(isfinite(points),'all'), ...
        'select_capture_marker_frame:nonfinite','Valid marker coordinates must be finite real values.');
end
