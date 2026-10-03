function bounds = gs3dx_scene_bounds(solids)
%GS3DX_SCENE_BOUNDS Fixed animation bounds containing every solved mesh.
    arguments
        solids (1,:) struct
    end
    lo = inf(3, 1);
    hi = -inf(3, 1);
    for i = 1:numel(solids)
        frames = solids(i).vertices_world;
        for j = 1:numel(frames)
            vertices = frames{j};
            if isempty(vertices)
                continue;
            end
            assert(size(vertices, 1) == 3 && all(isfinite(vertices), 'all'), ...
                'gs3dx:scene_bounds', 'World mesh vertices must be finite 3xN');
            lo = min(lo, min(vertices, [], 2));
            hi = max(hi, max(vertices, [], 2));
        end
    end
    assert(all(isfinite(lo)) && all(isfinite(hi)), 'gs3dx:scene_bounds', 'Scene must contain solved vertices');
    padding = max(0.08, 0.06 * (hi - lo));
    bounds = [lo-padding, hi+padding];
end
