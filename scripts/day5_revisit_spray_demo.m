function summary = day5_revisit_spray_demo()
%DAY5_REVISIT_SPRAY_DEMO Plan revisit route and log spray latency.
%   summary = day5_revisit_spray_demo() loads the Day 4 treatment map,
%   generates a post-coverage revisit trajectory, simulates targeted spray
%   events, and computes detection-to-treatment latency for one controller.

cfg = perception_config();
cfg.enabled = true;
cfg.seed = 2026;
cfg.output_root = fullfile("results", "perception_logs", "day5_revisit_PID");

day4_path = fullfile("results", "perception_logs", "day4_mapping_PID", "day4_mapping_summary.mat");
if ~isfile(day4_path)
    error("day5_revisit_spray_demo:MissingDay4", ...
        "Missing Day 4 output: %s. Run day4_perception_mapping_demo first.", day4_path);
end

loaded = load(day4_path, "summary", "treatment", "frame_logs");
day4_summary = loaded.summary;
treatment = loaded.treatment;
frame_logs = loaded.frame_logs;

if ~isfield(treatment, "zones") || isempty(treatment.zones)
    treatment = treatment_map("zones", treatment, cfg);
end
if isempty(treatment.zones)
    error("day5_revisit_spray_demo:NoZones", ...
        "No treatment zones were available from Day 4.");
end

prepare_output_root(cfg.output_root);

base_trajectory = struct();
base_trajectory.position = day4_summary.state_log(:, 1:3);

plan = revisit_planner(base_trajectory, treatment, cfg);
if ~plan.enabled
    error("day5_revisit_spray_demo:PlannerDisabled", ...
        "Revisit planner did not produce an enabled plan.");
end

[treatment, spray_state, spray_events, latency_by_zone] = simulate_revisit(plan, treatment, day4_summary.duration_s, cfg);
summary = build_summary(cfg, day4_summary, treatment, plan, spray_events, latency_by_zone);

save(fullfile(cfg.output_root, "day5_revisit_summary.mat"), ...
    "summary", "plan", "treatment", "spray_state", "spray_events", "latency_by_zone");
write_text_summary(summary, fullfile(cfg.output_root, "day5_revisit_summary.txt"));
write_spray_events_csv(spray_events, fullfile(cfg.output_root, "spray_events.csv"));
plot_revisit_plan(treatment, day4_summary, plan, spray_events, cfg, ...
    fullfile(cfg.output_root, "day5_revisit_plan.png"));

metrics_input = struct("controller", "PID_nominal_day5", ...
                       "treatment", treatment, ...
                       "spray_events", spray_events, ...
                       "frame_log", frame_logs);
metrics = perception_metrics(metrics_input, cfg);
save(fullfile(cfg.output_root, "day5_metrics.mat"), "metrics");
writetable(metrics, fullfile(cfg.output_root, "day5_metrics.csv"));
plot_latency_comparison(metrics, fullfile(cfg.output_root, "day5_latency_comparison.png"));

fprintf("Day 5 revisit and spray demo complete.\n");
fprintf("Planned zones: %d\n", summary.planned_zones);
fprintf("Sprayed zones: %d\n", summary.sprayed_zones);
fprintf("Missed zones: %d\n", summary.missed_zones);
fprintf("Mean detection-to-treatment latency: %.2f s\n", summary.mean_latency_s);
fprintf("Revisit path length: %.2f m\n", summary.revisit_path_length_m);
fprintf("Output root: %s\n", cfg.output_root);
end

function [treatment, spray_state, spray_events, latency_by_zone] = simulate_revisit(plan, treatment, phase1_duration_s, cfg)
spray_state = [];
spray_events = struct("zone_id", {}, "time_s", {}, "position", {}, "latency_s", {}, "dwell_s", {});
latency_by_zone = nan(1, max([plan.zones.id]));

traj = plan.trajectory;
for k = 1:numel(traj.t)
    uav_state = struct();
    uav_state.position = traj.x_ref(k, 1:3)';
    t_abs = phase1_duration_s + traj.t(k);
    [treatment, spray_state, spray_events, latency_by_zone] = spray_executor( ...
        uav_state, treatment, plan, t_abs, spray_state, cfg);
end
end

function summary = build_summary(cfg, day4_summary, treatment, plan, spray_events, latency_by_zone)
latencies = [spray_events.latency_s];
latencies = latencies(isfinite(latencies));
planned_zone_ids = [plan.zones.id];
sprayed_zone_ids = unique([spray_events.zone_id]);
missed_zone_ids = setdiff(planned_zone_ids, sprayed_zone_ids);

summary = struct();
summary.controller = "PID_nominal_day5";
summary.phase1_duration_s = day4_summary.duration_s;
summary.planned_zones = numel(planned_zone_ids);
summary.sprayed_zones = numel(sprayed_zone_ids);
summary.missed_zones = numel(missed_zone_ids);
summary.missed_zone_ids = missed_zone_ids;
summary.zone_order = plan.zone_order;
summary.revisit_duration_s = plan.trajectory.t(end);
summary.revisit_path_length_m = plan.summary.path_length_m;
summary.revisit_waypoints = plan.waypoints;
summary.latency_by_zone = latency_by_zone;
summary.mean_latency_s = scalar_or_nan(@mean, latencies);
summary.median_latency_s = scalar_or_nan(@median, latencies);
summary.max_latency_s = scalar_or_nan(@max, latencies);
summary.min_latency_s = scalar_or_nan(@min, latencies);
summary.spray_radius_m = cfg.replanner.spray_radius_m;
summary.spray_dwell_s = cfg.replanner.spray_dwell_s;
summary.output_root = string(cfg.output_root);
summary.active_sprayed_cells = nnz(treatment.sprayed);
summary.spray_events = spray_events;
end

function value = scalar_or_nan(fun_handle, values)
if isempty(values)
    value = NaN;
else
    value = fun_handle(values);
end
end

function prepare_output_root(output_root)
if ~exist(output_root, "dir")
    mkdir(output_root);
end
delete(fullfile(output_root, "day5_*"));
csv_path = fullfile(output_root, "spray_events.csv");
if isfile(csv_path)
    delete(csv_path);
end
end

function write_text_summary(summary, path_out)
fid = fopen(path_out, "w");
if fid < 0
    error("day5_revisit_spray_demo:SummaryOpenFailed", ...
        "Could not open summary file: %s", path_out);
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, "Day 5 revisit and spray summary\n");
fprintf(fid, "Controller: %s\n", summary.controller);
fprintf(fid, "Phase 1 duration: %.2f s\n", summary.phase1_duration_s);
fprintf(fid, "Revisit duration: %.2f s\n", summary.revisit_duration_s);
fprintf(fid, "Revisit path length: %.2f m\n", summary.revisit_path_length_m);
fprintf(fid, "Planned zones: %d\n", summary.planned_zones);
fprintf(fid, "Sprayed zones: %d\n", summary.sprayed_zones);
fprintf(fid, "Missed zones: %d\n", summary.missed_zones);
fprintf(fid, "Zone order: %s\n", mat2str(summary.zone_order));
fprintf(fid, "Mean latency: %.2f s\n", summary.mean_latency_s);
fprintf(fid, "Median latency: %.2f s\n", summary.median_latency_s);
fprintf(fid, "Max latency: %.2f s\n", summary.max_latency_s);
fprintf(fid, "Spray radius: %.2f m\n", summary.spray_radius_m);
fprintf(fid, "Spray dwell: %.2f s\n", summary.spray_dwell_s);

for i = 1:numel(summary.spray_events)
    e = summary.spray_events(i);
    fprintf(fid, "Event %d: zone %d at %.2f s, latency %.2f s, position [%.2f %.2f %.2f]\n", ...
        i, e.zone_id, e.time_s, e.latency_s, e.position(1), e.position(2), e.position(3));
end
end

function write_spray_events_csv(spray_events, path_out)
if isempty(spray_events)
    event_table = table();
else
    zone_id = [spray_events.zone_id]';
    time_s = [spray_events.time_s]';
    latency_s = [spray_events.latency_s]';
    dwell_s = [spray_events.dwell_s]';
    positions = reshape([spray_events.position], 3, [])';
    x_m = positions(:, 1);
    y_m = positions(:, 2);
    z_m = positions(:, 3);
    event_table = table(zone_id, time_s, latency_s, dwell_s, x_m, y_m, z_m);
end
writetable(event_table, path_out);
end

function plot_revisit_plan(treatment, day4_summary, plan, spray_events, cfg, output_path)
prob = 1 ./ (1 + exp(-treatment.log_odds));
fig = figure("Visible", "off", "Color", "w");
imagesc([0, cfg.field.length_m], [0, cfg.field.width_m], prob);
set(gca, "YDir", "normal");
axis equal tight;
hold on;
plot(day4_summary.reference.position(:, 1), day4_summary.reference.position(:, 2), ...
    "w-", "LineWidth", 1.1);
plot(plan.trajectory.x_ref(:, 1), plan.trajectory.x_ref(:, 2), ...
    "c-", "LineWidth", 1.8);
if ~isempty(plan.zones)
    centroids = reshape([plan.zones.centroid_xy], 2, [])';
    plot(centroids(:, 1), centroids(:, 2), "rx", "MarkerSize", 10, "LineWidth", 2);
    for i = 1:numel(plan.zones)
        text(plan.zones(i).centroid_xy(1) + 0.1, plan.zones(i).centroid_xy(2), ...
            sprintf("Z%d", plan.zones(i).id), "Color", "w", "FontWeight", "bold");
    end
end
if ~isempty(spray_events)
    positions = reshape([spray_events.position], 3, [])';
    plot(positions(:, 1), positions(:, 2), "go", "MarkerFaceColor", "g", "MarkerSize", 6);
end
colorbar;
caxis([0, 1]);
xlabel("x world [m]");
ylabel("y world [m]");
title("Day 5 Revisit Route and Spray Events");
legend(["Phase 1 path", "Revisit path", "Zone centroid", "Spray event"], ...
       "Location", "southoutside", "Orientation", "horizontal");
out_dir = fileparts(output_path);
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
