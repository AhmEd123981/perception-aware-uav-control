function reference = generate_agricultural_trajectory(cfg, params, duration_s, dt)
%GENERATE_AGRICULTURAL_TRAJECTORY Lawnmower reference for 15 m x 10 m field.

if nargin < 3 || isempty(duration_s)
    duration_s = get_nested(cfg, ["trajectory", "duration_s"], 100.0);
end
if nargin < 4 || isempty(dt)
    dt = get_nested(cfg, ["trajectory", "dt"], 0.01);
end
if nargin < 2
    params = struct(); %#ok<NASGU>
end

t = (0:dt:duration_s)';
margin = 0.7;
x_min = margin;
x_max = cfg.field.length_m - margin;
y_min = margin;
y_max = cfg.field.width_m - margin;
altitude = cfg.field.spray_altitude_m;
speed = get_nested(cfg, ["trajectory", "speed_mps"], 1.3);
track_spacing = get_nested(cfg, ["trajectory", "track_spacing_m"], 1.75);

lane_y = y_min:track_spacing:y_max;
if lane_y(end) < y_max
    lane_y(end + 1) = y_max;
end

waypoints = zeros(0, 3);
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

reference = reference_from_waypoints(t, waypoints, speed);
reference.trajectory_type = "agricultural";
reference.track_spacing_m = track_spacing;
end

function reference = reference_from_waypoints(t, waypoints, speed)
s = zeros(size(waypoints, 1), 1);
for i = 2:size(waypoints, 1)
    s(i) = s(i - 1) + norm(waypoints(i, :) - waypoints(i - 1, :));
end
query_s = min(speed * t, s(end));
position = interp1(s, waypoints, query_s, "linear", "extrap");
velocity = zeros(size(position));
for axis = 1:3
    velocity(:, axis) = gradient(position(:, axis), t);
end
yaw = atan2(velocity(:, 2), velocity(:, 1));
yaw(~isfinite(yaw)) = 0;
x_ref = zeros(numel(t), 13);
x_ref(:, 1:3) = position;
x_ref(:, 4:6) = velocity;
x_ref(:, 7) = cos(yaw / 2);
x_ref(:, 10) = sin(yaw / 2);

reference = struct();
reference.t = t;
reference.x_ref = x_ref;
reference.position = position;
reference.velocity = velocity;
reference.yaw = yaw;
reference.waypoints = waypoints;
end

function value = get_nested(s, names, default_value)
value = default_value;
cursor = s;
for i = 1:numel(names)
    name = char(names(i));
    if isstruct(cursor) && isfield(cursor, name)
        cursor = cursor.(name);
    else
        return;
    end
end
value = cursor;
end
