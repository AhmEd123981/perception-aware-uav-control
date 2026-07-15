function tests = test_spray_executor
%TEST_SPRAY_EXECUTOR Unit tests for targeted spray event logging.
tests = functiontests(localfunctions);
if nargout == 0
    results = run(tests);
    disp(results);
    assert(all([results.Passed]), "One or more spray_executor tests failed.");
    clear tests;
end
end

function testSprayEventTriggersAfterDwell(testCase)
cfg = perception_config();
cfg.replanner.spray_radius_m = 0.5;
cfg.replanner.spray_dwell_s = 1.0;
[treatment, plan] = fixture(cfg);
state = struct("position", [2; 2; 4]);
spray_state = [];
[treatment, spray_state, events] = spray_executor(state, treatment, plan, 5.0, spray_state, cfg);
verifyEmpty(testCase, events);
[~, ~, events, latency] = spray_executor(state, treatment, plan, 6.1, spray_state, cfg);
verifyEqual(testCase, numel(events), 1);
verifyEqual(testCase, events(1).latency_s, 4.1, "AbsTol", 1e-12);
verifyEqual(testCase, latency(1), 4.1, "AbsTol", 1e-12);
end

function testNoSprayWhenPassingTooQuickly(testCase)
cfg = perception_config();
cfg.replanner.spray_radius_m = 0.5;
cfg.replanner.spray_dwell_s = 1.0;
[treatment, plan] = fixture(cfg);
spray_state = [];
[treatment, spray_state, events] = spray_executor(struct("position", [2; 2; 4]), treatment, plan, 5.0, spray_state, cfg);
verifyEmpty(testCase, events);
[~, ~, events] = spray_executor(struct("position", [3; 3; 4]), treatment, plan, 5.5, spray_state, cfg);
verifyEmpty(testCase, events);
end

function testLatencyEqualsSprayMinusFirstDetection(testCase)
cfg = perception_config();
cfg.replanner.spray_radius_m = 0.5;
cfg.replanner.spray_dwell_s = 0.1;
[treatment, plan] = fixture(cfg);
state = struct("position", [2; 2; 4]);
spray_state = [];
[treatment, spray_state] = spray_executor(state, treatment, plan, 10.0, spray_state, cfg);
[~, ~, events] = spray_executor(state, treatment, plan, 10.2, spray_state, cfg);
verifyEqual(testCase, events(1).latency_s, 8.2, "AbsTol", 1e-12);
end

function [treatment, plan] = fixture(cfg)
treatment = treatment_map("init", [], cfg);
zone = struct("id", 1, "centroid_xy", [2 2], "area_m2", 1.0, ...
              "probability", 0.9, "first_detection_time", 2.0);
plan = struct();
plan.zones = zone;
plan.zone_order = 1;
plan.enabled = true;
end
