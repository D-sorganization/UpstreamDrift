function bounds = gs3dx_scene_bounds(solids, markers)
%GS3DX_SCENE_BOUNDS Fixed animation bounds containing solved meshes and measured markers.
    arguments
        solids (1,:) struct
        markers double = []
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
    if ~isempty(markers)
        if size(markers,1)~=3 && size(markers,2)==3,markers=permute(markers,[2 1 3]);end
        assert(isreal(markers) && size(markers,1)==3 && ndims(markers)<=3, ...
            'gs3dx:scene_bounds','World markers must be 3xNxF or Nx3xF');
        cloud=reshape(markers,3,[]);cloud=cloud(:,all(isfinite(cloud),1));
        if ~isempty(cloud)
            lo=min(lo,min(cloud,[],2));hi=max(hi,max(cloud,[],2));
        end
    end
    padding = max(0.08, 0.06 * (hi - lo));
    bounds = [lo-padding, hi+padding];
end
