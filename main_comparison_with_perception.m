function summary = main_comparison_with_perception(trajectory_name, varargin)
%MAIN_COMPARISON_WITH_PERCEPTION Run original controllers with perception.
%   summary = main_comparison_with_perception("agricultural") validates and
%   calls the original controller, dynamics, trajectory, and feedforward
%   functions, inserts run_perception_loop after every dynamics step, plans a
%   revisit route, logs spray events, and writes per-controller metrics.
%
%   This integration entry point never substitutes controller-response
%   approximations. If the original thesis API is not on the MATLAB path, it
%   fails before simulation and prints the missing files/functions.

if nargin < 1 || isempty(trajectory_name)
    trajectory_name = "agricultural";
end
trajectory_name = lower(string(trajectory_name));
if ~ismember(trajectory_name, ["agricultural", "circular"])
    error("main_comparison_with_perception:BadTrajectory", ...
        "trajectory_name must be 'agricultural' or 'circular'.");
end

opts = parse_options(varargin{:});
addpath(genpath(pwd));

cfg = perception_config();
cfg.enabled = true;
cfg.seed = opts.Seed;
cfg.camera.fps = opts.CameraFps;
cfg.output_root = fullfile(opts.OutputRoot, char(trajectory_name));
cfg.model.input_size = [opts.ImageSize, opts.ImageSize];
cfg.model.max_projected_pixels = opts.MaxProjectedPixels;
cfg.map.min_probability_for_zone = opts.MinZoneProbability;
cfg.replanner.enabled = opts.EnableRevisit;
cfg.dt = opts.Dt;

if ~isfolder(cfg.output_root)
    mkdir(cfg.output_root);
end

controller_names = ["PID", "LQR", "Hinf", "MPC", "SMC"];
controller_specs = controller_adapter("validate", controller_names);
api = resolve_project_api(trajectory_name);

params = load_project_params(opts.Dt);
options = load_controller_options();
math = build_math_context();
field_model = build_field_model();
reference = generate_reference(api, trajectory_name, cfg, params, opts);
print_spacing_diagnostics(reference, cfg);

fprintf("Existing project API selected for full perception integration:\n");
fprintf("  trajectory:  %s\n", api.trajectory_function);
fprintf("  feedforward: %s\n", api.feedforward_function);
fprintf("  dynamics:    %s\n", api.dynamics_function);
fprintf("  wind seed:   %d\n", opts.WindSeed);
fprintf("  initial condition seed: %d\n", opts.InitialConditionSeed);

run_results = repmat(empty_run_result(), numel(controller_specs), 1);
shared_initial_uav = initialize_uav_from_reference(reference, opts);

shared_worker = [];
try
    if string(cfg.model.runtime) == "python_onnx_worker"
        worker_cfg = cfg;
        worker_cfg.output_root = fullfile(opts.OutputRoot, char(trajectory_name), "shared_worker");
        shared_worker = onnx_worker("start", worker_cfg);
    end

    for i = 1:numel(controller_specs)
        spec = controller_specs(i);
        run_cfg = cfg;
        run_cfg.output_root = fullfile(opts.OutputRoot, char(trajectory_name), spec.name);
        prepare_run_dir(run_cfg.output_root);

        fprintf("\nFull integration controller run: %s / %s\n", spec.name, trajectory_name);
        fprintf("  calling init:    %s(params, dt, options)\n", spec.init_function);
        fprintf("  calling compute: %s(uav, x_ref, ctrl, params, math)\n", spec.compute_function);

        ctrl = controller_adapter("init", spec, params, opts.Dt, options);
        run_results(i) = run_controller_with_perception( ...
            spec, ctrl, api, shared_initial_uav, reference, field_model, ...
            params, math, run_cfg, opts);
    end
catch err
    if ~isempty(shared_worker)
        onnx_worker("stop", shared_worker);
    end
    rethrow(err);
end

metrics_input = repmat(struct("controller", "", "treatment", [], "spray_events", [], ...
    "frame_log", [], "field_model", field_model), numel(run_results), 1);
for i = 1:numel(run_results)
    metrics_input(i).controller = run_results(i).controller;
    metrics_input(i).treatment = run_results(i).treatment;
    metrics_input(i).spray_events = run_results(i).spray_events;
    metrics_input(i).frame_log = run_results(i).frame_logs;
    metrics_input(i).field_model = field_model;
end

metrics = perception_metrics(metrics_input, cfg);
combined_csv = fullfile(opts.OutputRoot, char(trajectory_name), "full_integration_metrics.csv");
writetable(metrics, combined_csv);
summary_txt = fullfile(opts.OutputRoot, char(trajectory_name), "full_integration_summary.txt");
write_full_integration_summary(summary_txt, trajectory_name, metrics, controller_specs, api, cfg);
save(fullfile(opts.OutputRoot, char(trajectory_name), "full_integration_summary.mat"), ...
    "run_results", "metrics", "cfg", "field_model", "reference");
plot_latency_comparison(metrics, fullfile(opts.OutputRoot, char(trajectory_name), "latency_comparison.png"));
if ~isempty(shared_worker)
    onnx_worker("stop", shared_worker);
end

summary = struct();
summary.trajectory = trajectory_name;
summary.output_root = string(fullfile(opts.OutputRoot, char(trajectory_name)));
summary.metrics_csv = string(combined_csv);
summary.summary_text = string(summary_txt);
summary.metrics = metrics;
summary.controller_files = controller_specs;
summary.api = api;

fprintf("\nFull integration complete for %s.\n", trajectory_name);
fprintf("Combined metrics: %s\n", combined_csv);
end

function write_full_integration_summary(path_out, trajectory_name, metrics, controller_specs, api, cfg)
fid = fopen(path_out, "w");
if fid < 0
    error("main_comparison_with_perception:SummaryOpenFailed", ...
        "Could not write integration summary: %s", path_out);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "Full perception integration summary\n");
fprintf(fid, "Trajectory: %s\n", trajectory_name);
fprintf(fid, "Camera FPS: %.2f\n", cfg.camera.fps);
fprintf(fid, "Treatment grid: %.2f m cells over %.1f m x %.1f m\n", ...
    cfg.map.resolution_m, cfg.field.length_m, cfg.field.width_m);
fprintf(fid, "Trajectory function: %s\n", api.trajectory_function);
fprintf(fid, "Feedforward function: %s\n", api.feedforward_function);
fprintf(fid, "Dynamics function: %s\n", api.dynamics_function);
fprintf(fid, "Controller files called:\n");
for i = 1:numel(controller_specs)
    fprintf(fid, "  %s: %s, %s\n", controller_specs(i).name, ...
        controller_specs(i).init_file, controller_specs(i).compute_file);
end
fprintf(fid, "\nController metrics:\n");
for i = 1:height(metrics)
    fprintf(fid, "%s | coverage %.2f %% | recall %.3f | false alarm %.5f m2/m2 | mean latency %.2f s | mean inference %.2f ms\n", ...
        metrics.controller(i), metrics.coverage_percent(i), metrics.affected_zone_recall(i), ...
        metrics.false_alarm_rate_per_m2(i), metrics.mean_latency_s(i), metrics.mean_inference_time_ms(i));
end
end

function opts = parse_options(varargin)
parser = inputParser();
parser.addParameter("Seed", 2026);
parser.addParameter("WindSeed", 2026);
parser.addParameter("InitialConditionSeed", 2026);
parser.addParameter("Duration", 100.0);
parser.addParameter("Dt", 0.01);
parser.addParameter("CameraFps", 5.0);
parser.addParameter("ImageSize", 256);  % av_segmentation_2020.onnx has fixed 256x256 input (was 512 for the old synthetic model)
parser.addParameter("MaxProjectedPixels", 20000);
parser.addParameter("MinZoneProbability", 0.55);
parser.addParameter("EnableRevisit", true);
parser.addParameter("OutputRoot", fullfile("results", "perception_logs", "full_integration"));
parser.parse(varargin{:});
opts = parser.Results;
end

function api = resolve_project_api(trajectory_name)
api = struct();
if trajectory_name == "agricultural"
    api.trajectory_function = resolve_function(["generate_agricultural_trajectory", ...
        "agricultural_trajectory", "trajectory_agricultural", "generate_lawnmower_trajectory"], ...
        "agricultural trajectory generator");
else
    api.trajectory_function = resolve_function(["generate_circular_trajectory", ...
        "circular_trajectory", "trajectory_circular"], "circular trajectory generator");
end
api.feedforward_function = resolve_function("compute_trajectory_feedforward", ...
    "trajectory feedforward function");
api.dynamics_function = resolve_function(["quadrotor_dynamics_step", ...
    "uav_dynamics_step", "dynamics_step", "simulate_uav_step", "quadrotor_dynamics"], ...
    "quadrotor dynamics step function");
end

function function_name = resolve_function(candidates, label)
candidates = string(candidates);
for i = 1:numel(candidates)
    if exist(candidates(i), "file") == 2
        function_name = char(candidates(i));
        return;
    end
end
message = sprintf("  - %s\n", strjoin(candidates, string(newline) + "  - "));
error("main_comparison_with_perception:MissingProjectAPI", ...
    "Missing original thesis %s. Tried:\n%s", label, message);
end

function params = load_project_params(dt)
candidate_functions = ["quadrotor_params", "uav_params", "default_quadrotor_params", ...
    "get_quadrotor_params", "init_params"];
params = struct();
for i = 1:numel(candidate_functions)
    if exist(candidate_functions(i), "file") == 2
        params = feval(candidate_functions(i));
        break;
    end
end
params = fill_missing_field(params, "m", 1.30);
params = fill_missing_field(params, "mass", 1.30);
params = fill_missing_field(params, "g", 9.81);
params = fill_missing_field(params, "dt", dt);
params = fill_missing_field(params, "Jxx", 0.0123);
params = fill_missing_field(params, "Jyy", 0.0123);
params = fill_missing_field(params, "Jzz", 0.0224);
if ~isfield(params, "J")
    params.J = diag([params.Jxx, params.Jyy, params.Jzz]);
end
if ~isfield(params, "I")
    params.I = params.J;
end
params = fill_missing_field(params, "Ix", params.Jxx);
params = fill_missing_field(params, "Iy", params.Jyy);
params = fill_missing_field(params, "Iz", params.Jzz);
params = fill_missing_field(params, "thrust_min", 0.0);
params = fill_missing_field(params, "thrust_max", 2.4 * params.m * params.g);
params = fill_missing_field(params, "torque_max", 0.5);
end

function params = fill_missing_field(params, name, value)
if ~isfield(params, name)
    params.(name) = value;
end
end

function options = load_controller_options()
options = struct();
options.verbose = false;
end

function math = build_math_context()
math = struct();
if exist("quat_to_dcm", "file") == 2
    math.quat_to_dcm = @quat_to_dcm;
else
    math.quat_to_dcm = @quat_to_dcm_local;
end
math.dcm_to_euler = @dcm_to_euler_local;
math.euler_to_dcm = @euler_to_dcm_local;
end

function field_model = build_field_model()
field_model = struct();
field_model.row_period_m = 0.75;
field_model.row_width_m = 0.35;

% Real-field mode: if a real Agriculture-Vision field texture AND its aligned
% ground-truth mask are present, the drone flies over REAL imagery
% (render_field auto-switches to orthomosaic sampling) and metrics score the
% detections against the REAL GT mask instead of synthetic ellipses. The
% texture and mask come from the same stitched AV test patches (see
% deep_learning/av_field_flyover.py --save-field) so they are pixel aligned.
% field_mask.png stores av 0-based ids (0 background, 1 drydown, 2 weed_cluster,
% 3 double_plant); perception_metrics converts to MATLAB ids (+1) and treats
% cfg.model.affected_class_ids = [2 3 4] as affected, matching run_perception_loop.
tex_path = fullfile("datasets", "real_field", "field_texture.png");
mask_path = fullfile("datasets", "real_field", "field_mask.png");
if isfile(tex_path) && isfile(mask_path)
    field_model.texture_rgb = imread(tex_path);
    gt_mask = imread(mask_path);
    if ndims(gt_mask) > 2
        gt_mask = gt_mask(:, :, 1);
    end
    field_model.texture_mask = uint8(gt_mask);
    field_model.texture_bounds = [0.0, 15.0; 0.0, 10.0];  % [xmin xmax; ymin ymax] meters
    % No synthetic affected_patches: the real GT mask is now the ground truth.
else
    % Fallback (no real field available): synthetic procedural field whose
    % ground truth is the analytic set of affected ellipses.
    field_model.affected_patches = struct( ...
        "center", {[5.8, 1.05], [10.5, 1.05], [4.5, 3.05], [11.5, 5.1], [7.5, 8.0]}, ...
        "radius", {[0.90, 0.45], [0.95, 0.50], [0.90, 0.65], [1.00, 0.80], [1.10, 0.55]}, ...
        "class_id", {3, 4, 5, 3, 4});
end
end

function reference = generate_reference(api, trajectory_name, cfg, params, opts)
cfg.trajectory = struct();
cfg.trajectory.name = char(trajectory_name);
cfg.trajectory.duration_s = opts.Duration;
cfg.trajectory.dt = opts.Dt;
cfg.trajectory.track_spacing_m = recommended_track_spacing(cfg);
cfg.trajectory.speed_mps = 1.3;
cfg.trajectory.seed = opts.Seed;

call_variants = {
    {cfg, params, opts.Duration, opts.Dt}
    {cfg, params}
    {params, opts.Dt, opts.Duration}
    {opts.Duration, opts.Dt, cfg}
    {opts.Duration, opts.Dt}
    {}
};
raw = [];
last_error = [];
for i = 1:numel(call_variants)
    try
        raw = feval(api.trajectory_function, call_variants{i}{:});
        last_error = [];
        break;
    catch err
        last_error = err;
    end
end
if ~isempty(last_error)
    rethrow(last_error);
end

reference = normalize_reference(raw, opts);
reference = apply_feedforward(api.feedforward_function, reference, params, opts.Dt);
end

function spacing = recommended_track_spacing(cfg)
[~, footprint_y] = nominal_camera_footprint(cfg);
spacing = min(1.75, 0.7 * footprint_y);
end

function reference = normalize_reference(raw, opts)
if isstruct(raw)
    reference = raw;
else
    error("main_comparison_with_perception:BadReference", ...
        "Trajectory generator must return a struct.");
end

if ~isfield(reference, "t") || isempty(reference.t)
    reference.t = (0:opts.Dt:opts.Duration)';
end
reference.t = reference.t(:);

if ~isfield(reference, "x_ref") || isempty(reference.x_ref)
    if isfield(reference, "position")
        position = reference.position;
    elseif isfield(reference, "x") && isfield(reference, "y") && isfield(reference, "z")
        position = [reference.x(:), reference.y(:), reference.z(:)];
    else
        error("main_comparison_with_perception:ReferenceMissingPosition", ...
            "Reference must contain x_ref, position, or x/y/z fields.");
    end
    if size(position, 1) ~= numel(reference.t)
        position = interp1(linspace(reference.t(1), reference.t(end), size(position, 1))', ...
            position, reference.t, "linear", "extrap");
    end
    velocity = zeros(size(position));
    for axis = 1:3
        velocity(:, axis) = gradient(position(:, axis), reference.t);
    end
    reference.x_ref = zeros(numel(reference.t), 13);
    reference.x_ref(:, 1:3) = position;
    reference.x_ref(:, 4:6) = velocity;
    reference.x_ref(:, 7) = 1.0;
end
end

function reference = apply_feedforward(feedforward_function, reference, params, dt)
call_variants = {
    {reference, params, dt}
    {reference, params}
    {reference}
};
for i = 1:numel(call_variants)
    try
        updated = feval(feedforward_function, call_variants{i}{:});
        if isstruct(updated)
            reference = updated;
        end
        return;
    catch
    end
end
end

function print_spacing_diagnostics(reference, cfg)
[footprint_x, footprint_y] = nominal_camera_footprint(cfg);
spacing = estimate_track_spacing(reference);
fprintf("Camera footprint at %.2f m: width_x=%.2f m, width_y=%.2f m\n", ...
    cfg.field.spray_altitude_m, footprint_x, footprint_y);
if isfinite(spacing)
    fprintf("Trajectory cross-track spacing estimate: %.2f m\n", spacing);
    fprintf("Recommended maximum spacing: %.2f m\n", 0.7 * footprint_y);
    if spacing > 0.7 * footprint_y
        warning("main_comparison_with_perception:WideTrackSpacing", ...
            "Track spacing %.2f m exceeds 70 percent of footprint width %.2f m. " + ...
            "Set the trajectory generator spacing near %.2f m for 70-95 percent coverage.", ...
            spacing, footprint_y, 0.7 * footprint_y);
    end
else
    fprintf("Trajectory cross-track spacing estimate: unavailable from reference.\n");
    fprintf("Recommended agricultural spacing: %.2f m\n", 0.7 * footprint_y);
end
end

function [footprint_x, footprint_y] = nominal_camera_footprint(cfg)
height_m = cfg.field.spray_altitude_m - cfg.field.ground_z_m;
horizontal_fov_rad = deg2rad(cfg.camera.horizontal_fov_deg);
image_h = cfg.camera.resolution(1);
image_w = cfg.camera.resolution(2);
vertical_fov_rad = 2 * atan(tan(horizontal_fov_rad / 2) * image_h / image_w);
footprint_x = 2 * height_m * tan(horizontal_fov_rad / 2);
footprint_y = 2 * height_m * tan(vertical_fov_rad / 2);
end

function spacing = estimate_track_spacing(reference)
spacing = NaN;
if isfield(reference, "waypoints") && size(reference.waypoints, 2) >= 2
    y_values = unique(round(reference.waypoints(:, 2), 3));
elseif isfield(reference, "x_ref") && size(reference.x_ref, 2) >= 2
    y_values = unique(round(reference.x_ref(:, 2), 2));
else
    return;
end
if numel(y_values) >= 2
    diffs = diff(sort(y_values));
    diffs = diffs(diffs > 0.25);
    if ~isempty(diffs)
        spacing = median(diffs);
    end
end
end

function result = run_controller_with_perception(spec, ctrl, api, initial_uav, reference, ...
    field_model, params, math, cfg, opts)
t_vec = reference.t(:);
t_vec = t_vec(t_vec <= opts.Duration + 1e-9);
uav = initial_uav;
treatment = treatment_map("init", [], cfg);
frame_idx = 1;
frame_logs = empty_frame_logs();
state_log = zeros(numel(t_vec), 13);
control_log = cell(numel(t_vec), 1);
wind_stream = RandStream("mt19937ar", "Seed", opts.WindSeed);

for k = 1:numel(t_vec)
    t = t_vec(k);
    x_ref = interpolate_reference(reference, t);
    [command, ctrl] = controller_adapter("compute", spec, uav, x_ref, ctrl, params, math);
    wind = sample_wind(wind_stream);
    uav = step_dynamics(api.dynamics_function, uav, command, params, opts.Dt, wind, t);
    uav_perception = normalize_uav_for_perception(uav, x_ref);

    [treatment, frame_log] = run_perception_loop( ...
        uav_perception, field_model, treatment, t, frame_idx, cfg);
    if frame_log.captured
        frame_logs(end + 1) = frame_log; %#ok<AGROW>
        frame_idx = frame_log.next_frame_idx;
    end

    state_log(k, :) = uav_to_x_ref_like(uav_perception, x_ref);
    control_log{k} = command;
end

treatment = treatment_map("zones", treatment, cfg);
base_trajectory = struct("t", t_vec, "x_ref", state_log);
plan = revisit_planner(base_trajectory, treatment, cfg);
[treatment, spray_events, latency_by_zone, revisit_log] = run_revisit_phase( ...
    spec, ctrl, api, uav, plan, treatment, params, math, cfg, opts, t_vec(end));

metrics_payload = struct("controller", spec.name, "treatment", treatment, ...
    "spray_events", spray_events, "frame_log", frame_logs, "field_model", field_model);
case_metrics = perception_metrics(metrics_payload, cfg);

save(fullfile(cfg.output_root, "controller_run.mat"), ...
    "treatment", "frame_logs", "state_log", "control_log", "plan", ...
    "spray_events", "latency_by_zone", "revisit_log", "case_metrics", "cfg", "field_model");
writetable(case_metrics, fullfile(cfg.output_root, "controller_metrics.csv"));
write_inference_histogram(frame_logs, fullfile(cfg.output_root, "inference_time_histogram.png"));

result = empty_run_result();
result.controller = spec.name;
result.treatment = treatment;
result.frame_logs = frame_logs;
result.plan = plan;
result.spray_events = spray_events;
result.latency_by_zone = latency_by_zone;
result.revisit_log = revisit_log;
result.metrics = case_metrics;
end

function [treatment, spray_events, latency_by_zone, revisit_log] = run_revisit_phase( ...
    spec, ctrl, api, uav, plan, treatment, params, math, cfg, opts, phase1_end_time)
spray_events = struct("zone_id", {}, "time_s", {}, "position", {}, "latency_s", {}, "dwell_s", {});
latency_by_zone = nan(1, 0);
revisit_log = zeros(0, 13);
if ~isfield(plan, "enabled") || ~plan.enabled || isempty(plan.trajectory.t)
    return;
end

spray_state = [];
traj = plan.trajectory;
revisit_log = zeros(numel(traj.t), 13);
for k = 1:numel(traj.t)
    t_abs = phase1_end_time + traj.t(k);
    x_ref = traj.x_ref(k, :);
    [command, ctrl] = controller_adapter("compute", spec, uav, x_ref, ctrl, params, math);
    wind = sample_wind(RandStream("mt19937ar", "Seed", opts.WindSeed + 100000 + k));
    uav = step_dynamics(api.dynamics_function, uav, command, params, opts.Dt, wind, t_abs);
    uav_perception = normalize_uav_for_perception(uav, x_ref);
    [treatment, spray_state, spray_events, latency_by_zone] = spray_executor( ...
        uav_perception, treatment, plan, t_abs, spray_state, cfg);
    revisit_log(k, :) = uav_to_x_ref_like(uav_perception, x_ref);
end
end

function x_ref = interpolate_reference(reference, t)
x_ref = zeros(1, 13);
for col = 1:min(13, size(reference.x_ref, 2))
    x_ref(col) = interp1(reference.t, reference.x_ref(:, col), t, "linear", "extrap");
end
if norm(x_ref(7:10)) < 1e-9
    x_ref(7) = 1.0;
end
end

function wind = sample_wind(stream)
wind.mean = [0.3, 0.2, 0.05];
wind.std = [0.2, 0.15, 0.05];
wind.velocity_mps = wind.mean + wind.std .* randn(stream, 1, 3);
end

function uav_next = step_dynamics(function_name, uav, command, params, dt, wind, t)
call_variants = {
    {uav, command, params, dt, wind, t}
    {uav, command, params, dt, wind}
    {uav, command, params, dt}
    {uav, command, params}
};
last_error = [];
for i = 1:numel(call_variants)
    try
        raw = feval(function_name, call_variants{i}{:});
        uav_next = normalize_dynamics_output(raw, uav);
        return;
    catch err
        last_error = err;
    end
end
rethrow(last_error);
end

function uav_next = normalize_dynamics_output(raw, previous_uav)
if isstruct(raw)
    uav_next = raw;
elseif isnumeric(raw)
    uav_next = previous_uav;
    uav_next.x = raw(:);
    uav_next.position = raw(1:3);
else
    error("main_comparison_with_perception:BadDynamicsOutput", ...
        "Dynamics output must be a struct or numeric state vector.");
end
end

function uav = initialize_uav_from_reference(reference, opts)
rng(opts.InitialConditionSeed);
x0 = reference.x_ref(1, :);
perturb = [0.03 * randn(1, 2), 0.01 * randn(1, 1)];
uav = struct();
state13 = x0(:);
state13(1:3) = state13(1:3) + perturb(:);
uav.state13 = state13;
uav.x = state13(1:3);
uav.v = state13(4:6);
uav.R = quat_to_dcm_local(state13(7:10));
uav.W = state13(11:13);
uav.position = uav.x;
uav.velocity = uav.v;
uav.quaternion = state13(7:10);
uav.body_rates = uav.W;
uav.R_world_body = uav.R;
end

function uav_out = normalize_uav_for_perception(uav, x_ref)
uav_out = uav;
if ~isfield(uav_out, "position")
    if isfield(uav_out, "pos")
        uav_out.position = uav_out.pos(:);
    elseif isfield(uav_out, "x")
        uav_out.position = uav_out.x(1:3);
    else
        uav_out.position = x_ref(1:3)';
    end
end
if ~isfield(uav_out, "x") || numel(uav_out.x) ~= 3
    uav_out.x = uav_out.position(:);
end
if ~isfield(uav_out, "velocity")
    if isfield(uav_out, "x") && numel(uav_out.x) >= 6
        uav_out.velocity = uav_out.x(4:6);
    else
        uav_out.velocity = x_ref(4:6)';
    end
end
if ~isfield(uav_out, "v")
    uav_out.v = uav_out.velocity(:);
end
if ~isfield(uav_out, "R_world_body")
    if isfield(uav_out, "quaternion")
        quat = uav_out.quaternion(:);
    elseif isfield(uav_out, "x") && numel(uav_out.x) >= 10
        quat = uav_out.x(7:10);
    else
        quat = x_ref(7:10)';
    end
    uav_out.R_world_body = quat_to_dcm_local(quat);
end
if ~isfield(uav_out, "R")
    uav_out.R = uav_out.R_world_body;
end
if ~isfield(uav_out, "W")
    if isfield(uav_out, "body_rates")
        uav_out.W = uav_out.body_rates(:);
    elseif isfield(uav_out, "state13") && numel(uav_out.state13) >= 13
        uav_out.W = uav_out.state13(11:13);
    elseif isfield(uav_out, "x") && numel(uav_out.x) >= 13
        uav_out.W = uav_out.x(11:13);
    else
        uav_out.W = zeros(3, 1);
    end
end
end

function state = uav_to_x_ref_like(uav, fallback)
state = fallback;
state(1:3) = uav.position(:)';
state(4:6) = uav.velocity(:)';
if isfield(uav, "quaternion") && numel(uav.quaternion) >= 4
    state(7:10) = uav.quaternion(:)';
elseif isfield(uav, "x") && numel(uav.x) >= 10
    state(7:10) = uav.x(7:10)';
end
end

function euler = dcm_to_euler_local(R)
%DCM_TO_EULER_LOCAL Convert DCM to roll-pitch-yaw using ZYX convention.
pitch = asin(max(-1, min(1, -R(3,1))));
if abs(cos(pitch)) > 1e-9
    roll = atan2(R(3,2), R(3,3));
    yaw = atan2(R(2,1), R(1,1));
else
    roll = 0;
    yaw = atan2(-R(1,2), R(2,2));
end
euler = [roll; pitch; yaw];
end

function R = euler_to_dcm_local(euler)
%EULER_TO_DCM_LOCAL Convert roll-pitch-yaw to DCM using ZYX convention.
roll = euler(1); pitch = euler(2); yaw = euler(3);
cr = cos(roll); sr = sin(roll);
cp = cos(pitch); sp = sin(pitch);
cy = cos(yaw); sy = sin(yaw);
Rz = [cy, -sy, 0; sy, cy, 0; 0, 0, 1];
Ry = [cp, 0, sp; 0, 1, 0; -sp, 0, cp];
Rx = [1, 0, 0; 0, cr, -sr; 0, sr, cr];
R = Rz * Ry * Rx;
end

function R = quat_to_dcm_local(q)
q = q(:);
if numel(q) < 4 || norm(q) < 1e-9
    R = eye(3);
    return;
end
q = q ./ norm(q);
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1 - 2*(y^2 + z^2), 2*(x*y - z*w), 2*(x*z + y*w); ...
     2*(x*y + z*w), 1 - 2*(x^2 + z^2), 2*(y*z - x*w); ...
     2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x^2 + y^2)];
end

function logs = empty_frame_logs()
logs = struct("captured", {}, "frame_idx", {}, "next_frame_idx", {}, ...
    "timestamp", {}, "exchange_dir", {}, "image_path", {}, "mask_path", {}, ...
    "inference_time_ms", {}, "is_warmup", {}, "num_affected_pixels", {}, ...
    "num_projected_pixels", {}, "footprint_xy", {}, "command", {}, "command_output", {});
end

function result = empty_run_result()
result = struct("controller", "", "treatment", [], "frame_logs", [], ...
    "plan", [], "spray_events", [], "latency_by_zone", [], "revisit_log", [], "metrics", []);
end

function prepare_run_dir(path_in)
if ~isfolder(path_in)
    mkdir(path_in);
end
exchange_dir = fullfile(path_in, "exchange");
if isfolder(exchange_dir)
    clean_directory(exchange_dir);
else
    mkdir(exchange_dir);
end
end

function clean_directory(path_in)
items = dir(path_in);
for i = 1:numel(items)
    name = items(i).name;
    if strcmp(name, ".") || strcmp(name, "..")
        continue;
    end
    target = fullfile(path_in, name);
    if items(i).isdir
        rmdir(target, "s");
    else
        delete(target);
    end
end
end

function write_inference_histogram(frame_logs, output_path)
if isempty(frame_logs)
    return;
end
times = [frame_logs.inference_time_ms];
if isfield(frame_logs, "is_warmup")
    times = times(~[frame_logs.is_warmup]);
elseif numel(times) > 3
    times = times(4:end);
end
times = times(isfinite(times));
if isempty(times)
    return;
end
fig = figure("Visible", "off", "Color", "w");
histogram(times, max(5, min(30, round(sqrt(numel(times))))));
xlabel("Inference time excluding first 3 warm-up frames [ms]");
ylabel("Frame count");
title("Persistent ONNX Runtime Inference Timing");
exportgraphics(fig, output_path, "Resolution", 200);
close(fig);
end
