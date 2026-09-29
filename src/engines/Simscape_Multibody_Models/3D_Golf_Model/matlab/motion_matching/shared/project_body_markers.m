function markers = project_body_markers(origins, rotations, body_indices, offsets)
%PROJECT_BODY_MARKERS Project fixed body-local attachments for one native pose.
% Origins/offsets are metres; rotations(:,:,b) maps body b into world axes.
% Return one world XYZ row per attachment, preserving supplied marker order.
    arguments
        origins (:,3) double {mustBeReal, mustBeFinite, mustBeNonempty}
        rotations (3,3,:) double {mustBeReal, mustBeFinite}
        body_indices (:,1) double {mustBeInteger, mustBePositive}
        offsets (:,3) double {mustBeReal, mustBeFinite}
    end
    assert(numel(body_indices) == size(offsets,1), ...
        'project_body_markers:attachmentCount', 'Each marker needs one body and offset.');
    assert(size(rotations,3) == size(origins,1) && all(body_indices <= size(origins,1)), ...
        'project_body_markers:bodyIndex', 'Every body must have an origin and rotation.');
    tolerance = 1e-8;
    for b = 1:size(rotations,3)
        rotation = rotations(:,:,b);
        assert(norm(rotation'*rotation-eye(3),'fro') < tolerance && ...
            abs(det(rotation)-1) < tolerance, 'project_body_markers:rotation', ...
            'Body orientations must be proper rotation matrices.');
    end
    markers = zeros(size(offsets));
    for m = 1:numel(body_indices)
        b = body_indices(m);
        markers(m,:) = origins(b,:) + offsets(m,:)*rotations(:,:,b)';
    end
end
