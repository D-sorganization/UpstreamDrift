function [R, c] = pose_from_cluster(cluster_t, reference)
%POSE_FROM_CLUSTER  Per-frame rigid-body pose of a 3-marker cluster.
%
%   [R, C] = POSE_FROM_CLUSTER(CLUSTER_T, REFERENCE) solves the rigid-body
%   pose of a 3-marker cluster against a reference geometry using the
%   SVD-based Kabsch algorithm.  For every frame t, finds R(:,:,t) (proper
%   rotation, det == +1) and c(t,:) (centroid) such that
%
%       cluster_t(:,:,t) ~= R(:,:,t) * (REFERENCE - mean(REFERENCE))
%                          + c(t,:)
%
%   Inputs:
%     CLUSTER_T (M x 3 x N) -- cluster positions per frame; each slice is
%                              [marker; xyz] with M=3 markers.  NaN-rows
%                              produce NaN outputs for that frame.
%     REFERENCE (3 x 3)     -- cluster geometry at the reference frame.
%
%   Outputs:
%     R (3 x 3 x N) -- proper rotations (det == +1).
%     C (N x 3)     -- per-frame centroids.
    arguments
        cluster_t (:, 3, :) double
        reference (3, 3) double {mustBeFinite}
    end

    nFrames = size(cluster_t, 3);
    R = nan(3, 3, nFrames);
    c = nan(nFrames, 3);

    ref_centroid = mean(reference, 1);
    ref_centered = reference - ref_centroid;

    for k = 1:nFrames
        frame = cluster_t(:, :, k);
        if any(~isfinite(frame), "all")
            continue;
        end
        ck = mean(frame, 1);
        cur_centered = frame - ck;
        H = ref_centered.' * cur_centered;
        [U, ~, V] = svd(H);
        d = sign(det(V * U.'));
        if d == 0
            d = 1;
        end
        Rk = V * diag([1, 1, d]) * U.';
        R(:, :, k) = Rk;
        c(k, :) = ck;

        assert(abs(det(Rk) - 1) < 1e-9, ...
            "Postcondition: rotation must be proper (det == +1)");
    end
end
