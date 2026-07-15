function reference = generate_circular_trajectory(cfg, params, duration_s, dt)
%GENERATE_CIRCULAR_TRAJECTORY Circular reference over the agricultural field.

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
center = [cfg.field.length_m / 2, cfg.field.width_m / 2];
radius = min(cfg.field.length_m, cfg.field.width_m) * 0.32;
altitude = cfg.field.spray_altitude_m;
period = max(20.0, duration_s / 2);
omega = 2 * pi / period;

position = zeros(numel(t), 3);
position(:, 1) = center(1) + radius * cos(omega * t);
position(:, 2) = center(2) + radius * sin(omega * t);
position(:, 3) = altitude;

velocity = zeros(size(position));
velocity(:, 1) = -radius * omega * sin(omega * t);
velocity(:, 2) = radius * omega * cos(omega * t);
yaw = atan2(velocity(:, 2), velocity(:, 1));

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
reference.waypoints = position(1:max(1, round(numel(t) / 16)):end, :);
reference.trajectory_type = "circular";
reference.radius_m = radius;
reference.period_s = period;
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
