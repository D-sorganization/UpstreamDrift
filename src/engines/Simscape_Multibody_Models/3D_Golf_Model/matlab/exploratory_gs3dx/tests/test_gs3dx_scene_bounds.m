function tests = test_gs3dx_scene_bounds
    tests = functiontests(localfunctions);
end
function setupOnce(t)
    t.TestData.original_path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end
function teardownOnce(t)
    path(t.TestData.original_path);
end
function testWholeSwingFitsIncludingRaisedClub(t)
    a = [0 1; -1 0; -1 1];
    b = [-2 3; -3 2; 0 2.7];
    solid = struct('vertices_world', {{a, [], b}});
    bounds = gs3dx_scene_bounds(solid);
    all_vertices = [a b];
    verifyTrue(t, all(bounds(:,1) < min(all_vertices,[],2)));
    verifyTrue(t, all(bounds(:,2) > max(all_vertices,[],2)));
end
function testNonfiniteGeometryRejected(t)
    solid = struct('vertices_world', {{[NaN; 0; 1]}});
    verifyError(t, @() gs3dx_scene_bounds(solid), 'gs3dx:scene_bounds');
end
function testEmptySceneRejected(t)
    solid = struct('vertices_world', {{[]}});
    verifyError(t, @() gs3dx_scene_bounds(solid), 'gs3dx:scene_bounds');
end
