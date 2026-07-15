function summary = day4_perception_mapping_demo(duration_s)
%DAY4_PERCEPTION_MAPPING_DEMO Run MATLAB-to-Python perception mapping.
%   summary = day4_perception_mapping_demo() executes a deterministic
%   agricultural pass, calls Python ONNX Runtime through run_perception_loop,
%   projects affected pixels to the ground plane, and accumulates a treatment
%   map. Outputs are written under results/perception_logs/day4_mapping_PID.

if nargin < 1 || isempty(duration_s)
    duration_s = 65.0;
end

cfg = perception_config();
cfg.enabled = true;
cfg.seed = 2026;
cfg.save_frames = true;
cfg.save_masks = true;
cfg.output_root = fullfile("results", "perception_logs", "day4_mapping_PID");
cfg.model.input_size = [512, 512];
cfg.model.max_projected_pixels = 12000;

onnx_path = fullfile("deep_learning", "weights", "crop_segmentation.onnx");
if ~isfile(onnx_path)
    error("day4_perception_mapping_demo:MissingONNX", ...
        "Missing ONNX model: %s. Run Day 3 before Day 4.", onnx_path);
end

prepare_output_root(cfg.output_root);

dt = 0.01;
t_vec = (0:dt:duration_s)';
reference = build_day4_lawnmower_reference(t_vec, cfg);
fprintf("Day 4 lawnmower spacing: %.2f m; camera footprint y-width: %.2f m\n", ...
    reference.lane_spacing_m, reference.footprint_width_y_m);
field_model = build_day4_field_model();
treatment = treatment_map("init", [], cfg);

actual_position = reference.position(1, :);
actual_velocity = [0, 0, 0];
tau_pos = 0.35;
frame_idx = 1;
frame_logs = empty_frame_logs();
state_log = zeros(numel(t_vec), 13);

for k = 1:numel(t_vec)
    ref_pos = reference.position(k, :);
    ref_vel = reference.velocity(k, :);
    position_error = ref_pos - actual_position;
    actual_velocity = ref_vel + position_error / tau_pos;
    actual_position = actual_position + actual_velocity * dt;
    actual_position(3) = cfg.field.spray_altitude_m;

    yaw = reference.yaw(k);
    uav_state = struct();
    uav_state.position = actual_position(:);
    uav_state.R_world_body = yaw_to_dcm(yaw);
    uav_state.velocity = actual_velocity(:);

    state_log(k, 1:3) = actual_position;
    state_log(k, 4:6) = actual_velocity;
    state_log(k, 7:10) = [cos(yaw / 2), 0, 0, sin(yaw / 2)];

    [treatment, frame_log] = run_perception_loop( ...
        uav_state, field_model, treatment, t_vec(k), frame_idx, cfg);

    if frame_log.captured
        frame_logs(end + 1) = frame_log; %#ok<AGROW>
        frame_idx = frame_log.next_frame_idx;
    end
end

treatment = treatment_map("zones", treatment, cfg);
summary = build_summary(cfg, duration_s, dt, treatment, frame_logs, reference, state_log);

if ~exist(cfg.output_root, "dir")
    mkdir(cfg.output_root);
end
save(fullfile(cfg.output_root, "day4_mapping_summary.mat"), "summary", "treatment", "frame_logs");
write_text_summary(summary, fullfile(cfg.output_root, "day4_mapping_summary.txt"));
plot_treatment_map(treatment, reference, field_model, cfg, ...
    fullfile(cfg.output_root, "day4_treatment_map.png"));

fprintf("Day 4 perception mapping demo complete.\n");
fprintf("Captured frames: %d at %.1f Hz\n", summary.captured_frames, summary.camera_fps);
fprintf("Mean inference time: %.1f ms\n", summary.mean_inference_time_ms);
fprintf("Affected projected pixels: %d\n", summary.total_projected_pixels);
fprintf("Detected treatment zones: %d\n", summary.detected_zones);
fprintf("Observed coverage: %.1f %%\n", summary.observed_coverage_percent);
fprintf("Output root: %s\n", cfg.output_root);
end

function summary = build_summary(cfg, duration_s, dt, treatment, frame_logs, reference, state_log)
captured = numel(frame_logs);
inference_times = [frame_logs.inference_time_ms];
projected = [frame_logs.num_projected_pixels];
affected = [frame_logs.num_affected_pixels];
if isempty(inference_times)
    inference_times = NaN;
    projected = 0;
    affected = 0;
end

prob = 1 ./ (1 + exp(-treatment.log_odds));
active_cells = prob >= cfg.map.min_probability_for_zone;

summary = struct();
summary.controller = "PID_nominal_day4";
summary.trajectory = "agricultural_lawnmower_day4";
summary.duration_s = duration_s;
summary.dt = dt;
summary.camera_fps = cfg.camera.fps;
summary.model_path = string(cfg.model.onnx_path);
summary.model_input_size = cfg.model.input_size;
summary.captured_frames = captured;
summary.mean_inference_time_ms = mean(inference_times, "omitnan");
summary.max_inference_time_ms = max(inference_times, [], "omitnan");
summary.total_affected_pixels = sum(affected);
summary.total_projected_pixels = sum(projected);
summary.active_treatment_cells = nnz(active_cells);
if isfield(treatment, "observed")
    summary.observed_coverage_percent = 100 * nnz(treatment.observed) / numel(treatment.observed);
else
    summary.observed_coverage_percent = NaN;
end
summary.detected_zones = numel(treatment.zones);
summary.output_root = string(cfg.output_root);
summary.reference = reference;
summary.state_log = state_log;
if ~isempty(treatment.zones)
    summary.zone_centroids = reshape([treatment.zones.centroid_xy], 2, [])';
    summary.zone_areas_m2 = [treatment.zones.area_m2]';
else
    summary.zone_centroids = zeros(0, 2);
    summary.zone_areas_m2 = zeros(0, 1);
end
end

function field_model = build_day4_field_model()
field_model = struct();
field_model.row_period_m = 0.75;
field_model.row_width_m = 0.35;
field_model.affected_patches = struct( ...
    "center", {[5.8, 1.05], [10.5, 1.05], [4.5, 3.05], [11.5, 5.1]}, ...
    "radius", {[0.90, 0.45], [0.95, 0.50], [0.90, 0.65], [1.00, 0.80]}, ...
    "class_id", {3, 4, 5, 3});
end

function reference = build_day4_lawnmower_reference(t, cfg)
margin = 1.0;
x_min = margin;
x_max = cfg.field.length_m - margin;
y_min = margin;
y_max = cfg.field.width_m - margin;
[~, footprint_width_y_m] = nominal_camera_footprint(cfg);
lane_spacing = min(1.75, 0.7 * footprint_width_y_m);
speed = 1.3;
altitude = cfg.field.spray_altitude_m;

lane_y = y_min:lane_spacing:y_max;
waypoints = [];
for i = 1:numel(lane_y)
    if mod(i, 2) == 1
        segment = [x_min, lane_y(i), altitude; x_max, lane_y(i), altitude];
    else
        segment = [x_max, lane_y(i), altitude; x_min, lane_y(i), altitude];
    end
    if isempty(waypoints)
        waypoints = segment;
    else
        waypoints = [waypoints; segment]; %#ok<AGROW>
    end
end

path_s = zeros(size(waypoints, 1), 1);
for i = 2:size(waypoints, 1)
    path_s(i) = path_s(i - 1) + norm(waypoints(i, :) - waypoints(i - 1, :));
end
query_s = min(speed * t, path_s(end));

position = interp1(path_s, waypoints, query_s, "linear", "extrap");
velocity = zeros(size(position));
for axis = 1:3
    velocity(:, axis) = gradient(position(:, axis), t);
end
yaw = atan2(velocity(:, 2), velocity(:, 1));
yaw(~isfinite(yaw)) = 0.0;

reference = struct();
reference.t = t;
reference.position = position;
reference.velocity = velocity;
reference.yaw = yaw;
reference.waypoints = waypoints;
reference.speed_mps = speed;
reference.lane_spacing_m = lane_spacing;
reference.footprint_width_y_m = footprint_width_y_m;
end

function [footprint_width_x_m, footprint_width_y_m] = nominal_camera_footprint(cfg)
height_m = cfg.field.spray_altitude_m - cfg.field.ground_z_m;
horizontal_fov_rad = deg2rad(cfg.camera.horizontal_fov_deg);
image_h = cfg.camera.resolution(1);
image_w = cfg.camera.resolution(2);
vertical_fov_rad = 2 * atan(tan(horizontal_fov_rad / 2) * image_h / image_w);
footprint_width_x_m = 2 * height_m * tan(horizontal_fov_rad / 2);
footprint_width_y_m = 2 * height_m * tan(vertical_fov_rad / 2);
end

function R = yaw_to_dcm(yaw)
cy = cos(yaw);
sy = sin(yaw);
R = [cy, -sy, 0; sy, cy, 0; 0, 0, 1];
end

function logs = empty_frame_logs()
logs = struct("captured", {}, "frame_idx", {}, "next_frame_idx", {}, ...
    "timestamp", {}, "exchange_dir", {}, "image_path", {}, "mask_path", {}, ...
    "inference_time_ms", {}, "is_warmup", {}, "num_affected_pixels", {}, ...
    "num_projected_pixels", {}, "footprint_xy", {}, "command", {}, "command_output", {});
end

function prepare_output_root(output_root)
if ~exist(output_root, "dir")
    mkdir(output_root);
end
subdirs = ["frames", "exchange"];
for i = 1:numel(subdirs)
    target = fullfile(output_root, subdirs(i));
    if exist(target, "dir")
        delete(fullfile(target, "*"));
    else
        mkdir(target);
    end
end
delete(fullfile(output_root, "day4_*"));
end

function write_text_summary(summary, path_out)
fid = fopen(path_out, "w");
if fid < 0
    error("day4_perception_mapping_demo:SummaryOpenFailed", ...
        "Could not open summary file: %s", path_out);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "Day 4 perception mapping summary\n");
fprintf(fid, "Controller: %s\n", summary.controller);
fprintf(fid, "Trajectory: %s\n", summary.trajectory);
fprintf(fid, "Duration: %.2f s\n", summary.duration_s);
fprintf(fid, "Camera FPS: %.2f\n", summary.camera_fps);
fprintf(fid, "Captured frames: %d\n", summary.captured_frames);
fprintf(fid, "Mean inference time: %.2f ms\n", summary.mean_inference_time_ms);
fprintf(fid, "Max inference time: %.2f ms\n", summary.max_inference_time_ms);
fprintf(fid, "Total affected pixels: %d\n", summary.total_affected_pixels);
fprintf(fid, "Total projected pixels: %d\n", summary.total_projected_pixels);
fprintf(fid, "Active treatment cells: %d\n", summary.active_treatment_cells);
fprintf(fid, "Observed coverage: %.2f %%\n", summary.observed_coverage_percent);
fprintf(fid, "Detected zones: %d\n", summary.detected_zones);
if summary.detected_zones > 0
    for i = 1:summary.detected_zones
        fprintf(fid, "Zone %d centroid: [%.2f, %.2f], area %.2f m^2\n", ...
            i, summary.zone_centroids(i, 1), summary.zone_centroids(i, 2), ...
            summary.zone_areas_m2(i));
    end
end
end

function plot_treatment_map(treatment, reference, field_model, cfg, output_path)
prob = 1 ./ (1 + exp(-treatment.log_odds));
fig = figure("Visible", "off", "Color", "w");
imagesc([0, cfg.field.length_m], [0, cfg.field.width_m], prob);
set(gca, "YDir", "normal");
axis equal tight;
hold on;
plot(reference.position(:, 1), reference.position(:, 2), "w-", "LineWidth", 1.2);
for i = 1:numel(field_model.affected_patches)
    p = field_model.affected_patches(i);
    rectangle("Position", [p.center(1) - p.radius(1), p.center(2) - p.radius(2), ...
              2 * p.radius(1), 2 * p.radius(2)], ...
              "Curvature", [1, 1], "EdgeColor", [1, 0.8, 0], "LineWidth", 1.2);
end
if isfield(treatment, "zones") && ~isempty(treatment.zones)
    centroids = reshape([treatment.zones.centroid_xy], 2, [])';
    plot(centroids(:, 1), centroids(:, 2), "rx", "MarkerSize", 10, "LineWidth", 2);
end
colorbar;
caxis([0, 1]);
xlabel("x world [m]");
ylabel("y world [m]");
title("Day 4 Treatment Map from ONNX Inference");
out_dir = fileparts(output_path);
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
