function summary = day2_perception_capture_demo(duration_s)
%DAY2_PERCEPTION_CAPTURE_DEMO Run Day 2 camera integration smoke scenario.
%   summary = day2_perception_capture_demo() runs one deterministic
%   lawnmower-style UAV pass with a PID-like nominal tracking response,
%   captures synthetic downward RGB frames at 5 Hz, and saves frame PNG files
%   plus pose metadata under results/perception_logs/day2_lawnmower_PID.
%
%   This script is intentionally separate from main_comparison_enhanced.m so
%   the perception stack can be validated without modifying the existing
%   five-controller comparison framework.

if nargin < 1 || isempty(duration_s)
    duration_s = 20.0;
end

cfg = perception_config();
cfg.enabled = true;
cfg.seed = 2026;
cfg.save_frames = true;
cfg.save_masks = false;
cfg.output_root = fullfile("results", "perception_logs", "day2_lawnmower_PID");

dt = 0.01;
frame_dt = 1.0 / cfg.camera.fps;
field_model = build_day2_field_model();
out_dir = fullfile(cfg.output_root, "frames");
prepare_output_directory(out_dir);

t = (0:dt:duration_s)';
reference = build_lawnmower_reference(t, cfg);
state_log = zeros(numel(t), 13);
frame_log = struct("frame_idx", {}, "timestamp", {}, "image_path", {}, ...
                   "metadata_path", {}, "position", {}, "world_footprint", {});

actual_position = reference.position(1, :);
actual_velocity = [0, 0, 0];
tau_pos = 0.35;
next_capture_time = 0.0;
frame_idx = 1;

for k = 1:numel(t)
    ref_pos = reference.position(k, :);
    ref_vel = reference.velocity(k, :);

    position_error = ref_pos - actual_position;
    actual_velocity = ref_vel + position_error / tau_pos;
    actual_position = actual_position + actual_velocity * dt;
    actual_position(3) = cfg.field.spray_altitude_m;

    yaw = reference.yaw(k);
    R_world_body = yaw_to_dcm(yaw);

    state_log(k, 1:3) = actual_position;
    state_log(k, 4:6) = actual_velocity;
    state_log(k, 7:10) = [cos(yaw / 2), 0, 0, sin(yaw / 2)];

    if t(k) + 1e-9 >= next_capture_time
        uav_state = struct();
        uav_state.position = actual_position(:);
        uav_state.R_world_body = R_world_body;
        uav_state.velocity = actual_velocity(:);

        frame = camera_simulator(uav_state, field_model, cfg, t(k), frame_idx);
        frame_log(end + 1) = struct("frame_idx", frame_idx, ...
                                    "timestamp", t(k), ...
                                    "image_path", string(frame.image_path), ...
                                    "metadata_path", string(frame.metadata_path), ...
                                    "position", actual_position, ...
                                    "world_footprint", frame.world_footprint); %#ok<AGROW>
        frame_idx = frame_idx + 1;
        next_capture_time = next_capture_time + frame_dt;
    end
end

summary = struct();
summary.controller = "PID_nominal_day2";
summary.trajectory = "agricultural_lawnmower_day2";
summary.duration_s = duration_s;
summary.dt = dt;
summary.camera_fps = cfg.camera.fps;
summary.expected_frames = floor(duration_s * cfg.camera.fps) + 1;
summary.captured_frames = numel(frame_log);
summary.output_root = string(cfg.output_root);
summary.frame_dir = string(out_dir);
summary.first_frame = frame_log(1).image_path;
summary.last_frame = frame_log(end).image_path;
summary.reference = reference;
summary.state_log = state_log;
summary.frame_log = frame_log;

summary_path = fullfile(cfg.output_root, "day2_capture_summary.mat");
if ~exist(cfg.output_root, "dir")
    mkdir(cfg.output_root);
end
save(summary_path, "summary");
write_text_summary(summary, fullfile(cfg.output_root, "day2_capture_summary.txt"));

fprintf("Day 2 perception capture demo complete.\n");
fprintf("Controller: %s\n", summary.controller);
fprintf("Trajectory: %s\n", summary.trajectory);
fprintf("Captured frames: %d / expected %d at %.1f Hz\n", ...
        summary.captured_frames, summary.expected_frames, summary.camera_fps);
fprintf("Frame directory: %s\n", out_dir);
fprintf("Summary: %s\n", summary_path);
end

function field_model = build_day2_field_model()
field_model = struct();
field_model.row_period_m = 0.75;
field_model.row_width_m = 0.35;
field_model.affected_patches = struct( ...
    "center", {[4.0, 2.8], [8.5, 6.6], [12.4, 4.2]}, ...
    "radius", {[0.85, 0.55], [1.05, 0.75], [0.65, 0.95]}, ...
    "class_id", {3, 4, 5});
end

function reference = build_lawnmower_reference(t, cfg)
margin = 1.0;
x_min = margin;
x_max = cfg.field.length_m - margin;
y_min = margin;
y_max = cfg.field.width_m - margin;
lane_spacing = 2.0;
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
end

function R = yaw_to_dcm(yaw)
cy = cos(yaw);
sy = sin(yaw);
R = [cy, -sy, 0; sy, cy, 0; 0, 0, 1];
end

function prepare_output_directory(out_dir)
if exist(out_dir, "dir")
    delete(fullfile(out_dir, "frame_*.png"));
    delete(fullfile(out_dir, "frame_*_pose.mat"));
else
    mkdir(out_dir);
end
end

function write_text_summary(summary, path_out)
fid = fopen(path_out, "w");
if fid < 0
    error("day2_perception_capture_demo:SummaryOpenFailed", ...
        "Could not open summary file: %s", path_out);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "Day 2 perception capture summary\n");
fprintf(fid, "Controller: %s\n", summary.controller);
fprintf(fid, "Trajectory: %s\n", summary.trajectory);
fprintf(fid, "Duration: %.2f s\n", summary.duration_s);
fprintf(fid, "Control dt: %.3f s\n", summary.dt);
fprintf(fid, "Camera FPS: %.2f\n", summary.camera_fps);
fprintf(fid, "Expected frames: %d\n", summary.expected_frames);
fprintf(fid, "Captured frames: %d\n", summary.captured_frames);
fprintf(fid, "Output root: %s\n", summary.output_root);
fprintf(fid, "Frame directory: %s\n", summary.frame_dir);
fprintf(fid, "First frame: %s\n", summary.first_frame);
fprintf(fid, "Last frame: %s\n", summary.last_frame);
end
