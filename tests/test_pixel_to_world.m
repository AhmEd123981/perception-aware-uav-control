function tests = test_pixel_to_world
%TEST_PIXEL_TO_WORLD Unit tests for flat-ground pixel projection.
tests = functiontests(localfunctions);
if nargout == 0
    results = run(tests);
    disp(results);
    assert(all([results.Passed]), "One or more pixel_to_world tests failed.");
    clear tests;
end
end

function testCenterPixelProjectsBelowCamera(testCase)
K = [100 0 50; 0 100 50; 0 0 1];
R_world_cam = diag([1, 1, -1]);
t_world_cam = [2; 3; 4];
uv = [50, 50];

xy = pixel_to_world(uv, K, R_world_cam, t_world_cam, 0);

verifyEqual(testCase, xy, [2, 3], "AbsTol", 1e-10);
end

function testTopLeftCornerProjectsKnownOffset(testCase)
K = [100 0 50; 0 100 50; 0 0 1];
R_world_cam = diag([1, 1, -1]);
t_world_cam = [0; 0; 4];
uv = [1, 1];
expected_xy = [4 * (1 - 50) / 100, 4 * (1 - 50) / 100];

xy = pixel_to_world(uv, K, R_world_cam, t_world_cam, 0);

verifyEqual(testCase, xy, expected_xy, "AbsTol", 1e-10);
end

function testOffCenterPixelProjectsExpectedX(testCase)
K = [100 0 50; 0 100 50; 0 0 1];
R_world_cam = diag([1, 1, -1]);
t_world_cam = [0; 0; 4];
uv = [75, 50];

xy = pixel_to_world(uv, K, R_world_cam, t_world_cam, 0);

verifyEqual(testCase, xy, [1, 0], "AbsTol", 1e-10);
end

function testParallelRayReturnsNaN(testCase)
K = [100 0 50; 0 100 50; 0 0 1];
R_world_cam = [0 0 1; 0 1 0; -1 0 0];
t_world_cam = [0; 0; 4];
uv = [50, 50];

xy = pixel_to_world(uv, K, R_world_cam, t_world_cam, 0);

verifyTrue(testCase, all(isnan(xy)));
end

function testBatchProjectionShape(testCase)
K = [100 0 50; 0 100 50; 0 0 1];
R_world_cam = diag([1, 1, -1]);
t_world_cam = [0; 0; 4];
uv = [50, 50; 60, 60; 40, 50];

xy = pixel_to_world(uv, K, R_world_cam, t_world_cam, 0);

verifySize(testCase, xy, [3, 2]);
verifyTrue(testCase, all(isfinite(xy), "all"));
end
