function tests = test_revisit_planner
%TEST_REVISIT_PLANNER Unit tests for affected-zone revisit planning.
tests = functiontests(localfunctions);
if nargout == 0
    results = run(tests);
    disp(results);
    assert(all([results.Passed]), "One or more revisit_planner tests failed.");
    clear tests;
end
end

function testNearestNeighborOrderWithKnownCentroids(testCase)
cfg = perception_config();
base = struct("position", [0 0 4]);
zones = struct("id", {1, 2, 3}, ...
               "centroid_xy", {[4 0], [1 0], [2 0]}, ...
               "area_m2", {1, 1, 1}, ...
               "probability", {0.9, 0.9, 0.9}, ...
               "first_detection_time", {0, 0, 0});
treatment = struct("zones", zones);
plan = revisit_planner(base, treatment, cfg);
verifyTrue(testCase, plan.enabled);
verifyEqual(testCase, plan.zone_order, [2 3 1]);
end

function testEmptyZoneListDisablesPlan(testCase)
cfg = perception_config();
base = struct("position", [0 0 4]);
treatment = struct("zones", struct([]));
plan = revisit_planner(base, treatment, cfg);
verifyFalse(testCase, plan.enabled);
verifyEqual(testCase, size(plan.trajectory.x_ref, 2), 13);
end

function testWaypointsUseConfiguredAltitude(testCase)
cfg = perception_config();
cfg.replanner.revisit_altitude_m = 3.5;
base = struct("position", [0 0 4]);
zones = struct("id", {1}, ...
               "centroid_xy", {[2 2]}, ...
               "area_m2", {1}, ...
               "probability", {0.9}, ...
               "first_detection_time", {0});
treatment = struct("zones", zones);
plan = revisit_planner(base, treatment, cfg);
verifyEqual(testCase, plan.waypoints(:, 3), cfg.replanner.revisit_altitude_m);
verifyEqual(testCase, plan.trajectory.x_ref(end, 3), cfg.replanner.revisit_altitude_m, "AbsTol", 1e-12);
end
