function plan = revisit_planner(base_trajectory, treatment_map_struct, cfg)
%REVISIT_PLANNER Build an affected-zone revisit trajectory.
%   The planner extracts treatment-zone centroids, orders them with a
%   nearest-neighbor TSP heuristic, and returns a time-parameterized
%   reference struct with t and x_ref fields. x_ref is Nx13:
%   position, velocity, quaternion [w x y z], and body rates.

arguments
    base_trajectory struct
    treatment_map_struct struct
    cfg struct
end

zones = treatment_map_struct.zones;
plan = struct();
plan.enabled = cfg.replanner.enabled && ~isempty(zones);
plan.zones = zones;
plan.waypoints = [];
plan.spray_events = struct("zone_id", {}, "time_s", {}, "position", {}, "latency_s", {});
plan.summary = struct("num_zones", numel(zones), "path_length_m", 0.0);

if ~plan.enabled
    plan.trajectory = empty_reference();
    return;
end

start_xy = infer_final_xy(base_trajectory);
start_z = infer_final_z(base_trajectory, cfg.replanner.revisit_altitude_m);
centroids = reshape([zones.centroid_xy], 2, [])';
order = nearest_neighbor_order(start_xy, centroids);
ordered = centroids(order, :);

alt = cfg.replanner.revisit_altitude_m;
speed = cfg.replanner.revisit_speed_mps;
waypoints = zeros(size(ordered, 1), 4);
waypoints(:, 1:2) = ordered;
waypoints(:, 3) = alt;
waypoints(:, 4) = speed;

plan.waypoints = waypoints;
plan.zone_order = [zones(order).id];
plan.summary.path_length_m = compute_path_length(start_xy, ordered);
plan.trajectory = waypoints_to_time_parameterized_reference(waypoints, [start_xy, start_z], cfg);
end

function xy = infer_final_xy(base_trajectory)
if isfield(base_trajectory, "position")
    pos = base_trajectory.position;
    xy = pos(end, 1:2);
elseif isfield(base_trajectory, "x_ref")
    xy = base_trajectory.x_ref(end, 1:2);
elseif isfield(base_trajectory, "x") && isfield(base_trajectory, "y")
    xy = [base_trajectory.x(end), base_trajectory.y(end)];
else
    xy = [0.0, 0.0];
end
end

function z = infer_final_z(base_trajectory, default_z)
if isfield(base_trajectory, "position")
    pos = base_trajectory.position;
    z = pos(end, 3);
elseif isfield(base_trajectory, "x_ref") && size(base_trajectory.x_ref, 2) >= 3
    z = base_trajectory.x_ref(end, 3);
elseif isfield(base_trajectory, "z")
    z = base_trajectory.z(end);
else
    z = default_z;
end
end

function order = nearest_neighbor_order(start_xy, points)
remaining = 1:size(points, 1);
current = start_xy;
order = zeros(1, numel(remaining));
for k = 1:numel(order)
    distances = vecnorm(points(remaining, :) - current, 2, 2);
    [~, idx] = min(distances);
    order(k) = remaining(idx);
    current = points(order(k), :);
    remaining(idx) = [];
end
end

function length_m = compute_path_length(start_xy, ordered)
if isempty(ordered)
    length_m = 0.0;
    return;
end
pts = [start_xy; ordered];
length_m = sum(vecnorm(diff(pts, 1, 1), 2, 2));
end

function traj = waypoints_to_time_parameterized_reference(waypoints, start_xyz, cfg)
dt = get_cfg_or(cfg, "dt", 0.01);
speed = max(cfg.replanner.revisit_speed_mps, 1e-3);
% Hold each zone for the spray dwell PLUS a settling margin so the controller
% can converge inside spray_radius before the spray-dwell clock must complete.
settle_s = 0.0;
if isfield(cfg.replanner, "spray_settle_s")
    settle_s = cfg.replanner.spray_settle_s;
end
dwell_s = max(cfg.replanner.spray_dwell_s + settle_s, 0.0);

points = [start_xyz; waypoints(:, 1:3)];
t_all = [];
p_all = [];
time_offset = 0.0;

for i = 1:(size(points, 1) - 1)
    p0 = points(i, :);
    p1 = points(i + 1, :);
    distance = norm(p1 - p0);
    duration = max(distance / speed, dt);
    local_t = (0:dt:duration)';
    if local_t(end) < duration
        local_t(end + 1, 1) = duration;
    end
    s = local_t ./ duration;
    sigma = 3 .* s.^2 - 2 .* s.^3;
    segment_pos = p0 + sigma .* (p1 - p0);

    if ~isempty(t_all)
        local_t = local_t(2:end);
        segment_pos = segment_pos(2:end, :);
    end
    t_all = [t_all; time_offset + local_t]; %#ok<AGROW>
    p_all = [p_all; segment_pos]; %#ok<AGROW>
    time_offset = t_all(end);

    if dwell_s > 0
        dwell_t = (dt:dt:dwell_s)';
        dwell_pos = repmat(p1, numel(dwell_t), 1);
        t_all = [t_all; time_offset + dwell_t]; %#ok<AGROW>
        p_all = [p_all; dwell_pos]; %#ok<AGROW>
        time_offset = t_all(end);
    end
end

if isempty(t_all)
    traj = empty_reference();
    return;
end

velocity = finite_difference_velocity(p_all, t_all);
x_ref = zeros(numel(t_all), 13);
x_ref(:, 1:3) = p_all;
x_ref(:, 4:6) = velocity;
x_ref(:, 7) = 1.0;

traj = struct();
traj.t = t_all;
traj.x_ref = x_ref;
traj.dt = dt;
traj.yaw_ref = zeros(numel(t_all), 1);
traj.waypoints = waypoints;
end

function velocity = finite_difference_velocity(position, t)
velocity = zeros(size(position));
if size(position, 1) < 2
    return;
end
for axis = 1:3
    velocity(:, axis) = gradient(position(:, axis), t);
end
end

function value = get_cfg_or(cfg, name, default_value)
if isfield(cfg, name)
    value = cfg.(name);
else
    value = default_value;
end
end

function traj = empty_reference()
traj = struct();
traj.t = zeros(0, 1);
traj.x_ref = zeros(0, 13);
traj.dt = 0.01;
traj.yaw_ref = zeros(0, 1);
traj.waypoints = zeros(0, 4);
end
