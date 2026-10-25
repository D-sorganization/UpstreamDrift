function points = gs3dx_foot_contacts(ankle, toe, foot_length, foot_width)
%GS3DX_FOOT_CONTACTS  Rectangular foot contact corners from ankle and toe (#11173).
%
%   POINTS = GS3DX_FOOT_CONTACTS(ANKLE, TOE, FOOT_LENGTH, FOOT_WIDTH) returns
%   3x4xN contact positions: heel medial, heel lateral, toe medial, toe lateral.
%   ANKLE and TOE are 3xN (m).  FOOT_LENGTH and FOOT_WIDTH are positive finite
%   scalars (m).  Row 3 of every contact equals the ankle z for that frame
%   (pass sole height as ankle z when contacts should lie on the sole plane).
%
%   u is the xy unit vector from ankle to toe; if that distance is under 1 cm,
%   u = [1; 0].  Heel xy = ankle_xy - u*(0.25*foot_length).  Toe xy =
%   ankle_xy + u*(0.75*foot_length).  v = [-u(2); u(1)].  Medial is
%   -v*(foot_width/2), lateral is +v*(foot_width/2).
%
%   Error id gs3dx:foot.

    assert(size(ankle, 1) == 3, 'gs3dx:foot', 'ANKLE must be 3xN');
    assert(size(toe, 1) == 3, 'gs3dx:foot', 'TOE must be 3xN');
    n = size(ankle, 2);
    assert(size(toe, 2) == n, 'gs3dx:foot', 'ANKLE and TOE must have the same column count');
    assert(isscalar(foot_length) && isfinite(foot_length) && foot_length > 0, ...
        'gs3dx:foot', 'FOOT_LENGTH must be a positive finite scalar');
    assert(isscalar(foot_width) && isfinite(foot_width) && foot_width > 0, ...
        'gs3dx:foot', 'FOOT_WIDTH must be a positive finite scalar');
    assert(all(isfinite(ankle(:))) && all(isfinite(toe(:))), ...
        'gs3dx:foot', 'ANKLE and TOE must be finite');

    points = zeros(3, 4, n);
    for k = 1:n
        ankle_xy = ankle(1:2, k);
        z = ankle(3, k);
        d = toe(1:2, k) - ankle_xy;
        if norm(d) < 0.01
            u = [1; 0];
        else
            u = d / norm(d);
        end
        v = [-u(2); u(1)];
        heel_xy = ankle_xy - u * (0.25 * foot_length);
        toe_xy = ankle_xy + u * (0.75 * foot_length);
        medial = -v * (foot_width / 2);
        lateral = v * (foot_width / 2);
        points(:, 1, k) = [heel_xy + medial; z];
        points(:, 2, k) = [heel_xy + lateral; z];
        points(:, 3, k) = [toe_xy + medial; z];
        points(:, 4, k) = [toe_xy + lateral; z];

        pts_xy = points(1:2, :, k);
        d01 = norm(pts_xy(:, 1) - pts_xy(:, 2));
        d03 = norm(pts_xy(:, 1) - pts_xy(:, 3));
        d12 = norm(pts_xy(:, 1) - pts_xy(:, 4));
        assert(abs(d01 - foot_width) <= 1e-9 && abs(d03 - foot_length) <= 1e-9, ...
            'gs3dx:foot', 'Contact corners must form a foot_length by foot_width rectangle');
        diag = norm(pts_xy(:, 1) - pts_xy(:, 4));
        assert(abs(diag - hypot(foot_length, foot_width)) <= 1e-9, ...
            'gs3dx:foot', 'Rectangle diagonal must match hypot(foot_length, foot_width)');
        centroid = mean(pts_xy, 2);
        want_centroid = ankle_xy + u * (0.25 * foot_length);
        assert(norm(centroid - want_centroid) <= 1e-9, 'gs3dx:foot', ...
            'Rectangle centroid xy must equal ankle_xy + u*(0.25*foot_length)');
    end
end
