function tests = test_treatment_map
%TEST_TREATMENT_MAP Unit tests for treatment map observation and zones.
tests = functiontests(localfunctions);
if nargout == 0
    results = run(tests);
    disp(results);
    assert(all([results.Passed]), "One or more treatment_map tests failed.");
    clear tests;
end
end

function testInitReturnsZeroLogOddsAndObservationCount(testCase)
cfg = perception_config();
map = treatment_map("init", [], cfg);
verifyEqual(testCase, map.log_odds, zeros(size(map.log_odds)));
verifyEqual(testCase, map.observation_count, zeros(size(map.observation_count)));
verifyFalse(testCase, any(map.observed, "all"));
end

function testUpdateMarksFootprintObserved(testCase)
cfg = perception_config();
map = treatment_map("init", [], cfg);
footprint = [0 0; 2 0; 2 2; 0 2];
map = treatment_map("update", map, zeros(0, 2), zeros(0, 5), 1.0, cfg, footprint);
verifyGreaterThan(testCase, nnz(map.observed), 0);
verifyGreaterThan(testCase, nnz(map.observation_count), 0);
end

function testExtractZonesRespectsMinimumArea(testCase)
cfg = perception_config();
cfg.map.min_zone_area_m2 = 0.50;
map = treatment_map("init", [], cfg);
class_probs = zeros(1, 5);
class_probs(3) = 1;
map = treatment_map("update", map, [1.0, 1.0], class_probs, 1.0, cfg, [0 0; 2 0; 2 2; 0 2]);
map = treatment_map("zones", map, cfg);
verifyEqual(testCase, numel(map.zones), 0);

pts = [];
for x = 3.0:0.25:4.0
    for y = 3.0:0.25:4.0
        pts(end + 1, :) = [x, y]; %#ok<AGROW>
    end
end
class_probs = zeros(size(pts, 1), 5);
class_probs(:, 3) = 1;
map = treatment_map("update", map, pts, class_probs, 2.0, cfg, [2.5 2.5; 4.5 2.5; 4.5 4.5; 2.5 4.5]);
map = treatment_map("zones", map, cfg);
verifyGreaterThanOrEqual(testCase, numel(map.zones), 1);
end

function testLogOddsSaturatesOnRepeatedHits(testCase)
cfg = perception_config();
cfg.map.log_odds_max = 2.0;
map = treatment_map("init", [], cfg);
class_probs = zeros(1, 5);
class_probs(3) = 1;
for i = 1:20
    map = treatment_map("update", map, [5.0, 5.0], class_probs, i, cfg, [4 4; 6 4; 6 6; 4 6]);
end
[row, col] = local_world_to_grid(map, [5.0, 5.0]);
verifyEqual(testCase, map.log_odds(row, col), cfg.map.log_odds_max, "AbsTol", 1e-12);
end

function [row, col] = local_world_to_grid(map, xy)
col = floor(xy(1) / map.resolution_m) + 1;
row = floor(xy(2) / map.resolution_m) + 1;
end
