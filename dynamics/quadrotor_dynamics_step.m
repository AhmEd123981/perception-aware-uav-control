function uav_next = quadrotor_dynamics_step(uav, command, params, dt, wind, t)
%QUADROTOR_DYNAMICS_STEP Step the quadrotor state for perception integration.
%   Supports both command conventions used in this thesis workspace:
%   1. Real thesis controllers: numeric [thrust; tau_x; tau_y; tau_z].
%   2. Perception fallback/demo commands: struct with accel_cmd.
%
%   The output keeps the real-controller fields (x, v, R, W) and the
%   perception fields (position, velocity, quaternion, R_world_body).

if nargin < 5 || isempty(wind)
    wind = struct("velocity_mps", [0, 0, 0]);
end
if nargin < 6
    t = 0; %#ok<NASGU>
end

state = normalize_state(uav);
wind_velocity = get_wind_velocity(wind);

if isnumeric(command) && numel(command) >= 4
    [position_next, velocity_next, R_next, W_next] = integrate_thrust_torque( ...
        state.position, state.velocity, state.R, state.W, command(:), params, dt, wind_velocity(:));
else
    [position_next, velocity_next, R_next, W_next] = integrate_accel_command( ...
        state.position, state.velocity, command, dt, wind_velocity(:));
end

position_next(3) = max(0.05, position_next(3));
quat = dcm_to_quat_local(R_next);

uav_next = uav;
uav_next.position = position_next;
uav_next.velocity = velocity_next;
uav_next.quaternion = quat;
uav_next.body_rates = W_next;
uav_next.R_world_body = R_next;
uav_next.x = position_next;
uav_next.v = velocity_next;
uav_next.R = R_next;
uav_next.W = W_next;
uav_next.state13 = [position_next; velocity_next; quat; W_next];
uav_next.command = command;
uav_next.params = params;
end

function [x_next, v_next, R_next, W_next] = integrate_accel_command(x, v, command, dt, wind_velocity)
accel_cmd = get_command_accel(command);
drag_gain = 0.18;
accel = accel_cmd(:) + drag_gain * (wind_velocity - v);
v_next = v + accel * dt;
x_next = x + v_next * dt;
yaw = get_command_yaw(command);
R_next = quat_to_dcm_local([cos(yaw / 2); 0; 0; sin(yaw / 2)]);
W_next = zeros(3, 1);
end

function [x_next, v_next, R_next, W_next] = integrate_thrust_torque(x, v, R, W, u, params, dt, wind_velocity)
thrust = u(1);
tau = u(2:4);
mass = get_mass(params);
gravity = get_gravity(params);
J = get_inertia(params);
d_force = mass * 0.18 * (wind_velocity - v);

[k1_v, k1_w] = derivatives(R, W, thrust, tau, mass, gravity, J, d_force);
[k2_v, k2_w] = derivatives(R, W + 0.5*dt*k1_w, thrust, tau, mass, gravity, J, d_force);
[k3_v, k3_w] = derivatives(R, W + 0.5*dt*k2_w, thrust, tau, mass, gravity, J, d_force);
[k4_v, k4_w] = derivatives(R, W + dt*k3_w, thrust, tau, mass, gravity, J, d_force);

v_next = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
W_next = W + (dt/6) * (k1_w + 2*k2_w + 2*k3_w + k4_w);
x_next = x + v_next * dt;

R_next = integrate_rotation(R, 0.5 * (W + W_next), dt);
[U, ~, V] = svd(R_next);
R_next = U * V';
if det(R_next) < 0
    U(:, 3) = -U(:, 3);
    R_next = U * V';
end

if norm(v_next) > 15
    v_next = v_next * (15 / max(norm(v_next), eps));
end
W_next = max(-8, min(8, W_next));
end

function [dv, dW] = derivatives(R, W, thrust, tau, mass, gravity, J, d_force)
dv = [0; 0; -gravity] + (thrust * R * [0; 0; 1] + d_force) / mass;
dW = J \ (tau - cross(W, J * W));
end

function R_next = integrate_rotation(R, omega, dt)
w_norm = norm(omega);
if w_norm > 1e-8
    theta = w_norm * dt;
    w_unit = omega / w_norm;
    K = skew(w_unit);
    R_next = R * (eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K));
else
    R_next = R * (eye(3) + skew(omega) * dt);
end
end

function S = skew(v)
S = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function state = normalize_state(uav)
state.position = get_vec(uav, "position", 1:3, [0; 0; 4]);
if isfield(uav, "x") && numel(uav.x) == 3
    state.position = uav.x(:);
end

state.velocity = get_vec(uav, "velocity", 4:6, [0; 0; 0]);
if isfield(uav, "v")
    state.velocity = uav.v(:);
end

if isfield(uav, "R")
    state.R = uav.R;
elseif isfield(uav, "R_world_body")
    state.R = uav.R_world_body;
elseif isfield(uav, "quaternion")
    state.R = quat_to_dcm_local(uav.quaternion);
elseif isfield(uav, "x") && numel(uav.x) >= 10
    state.R = quat_to_dcm_local(uav.x(7:10));
else
    state.R = eye(3);
end

if isfield(uav, "W")
    state.W = uav.W(:);
elseif isfield(uav, "body_rates")
    state.W = uav.body_rates(:);
elseif isfield(uav, "x") && numel(uav.x) >= 13
    state.W = uav.x(11:13);
else
    state.W = zeros(3, 1);
end
end

function value = get_vec(uav, field_name, idx, default_value)
if isfield(uav, field_name)
    value = uav.(field_name)(:);
elseif isfield(uav, "state13") && numel(uav.state13) >= max(idx)
    value = uav.state13(idx);
elseif isfield(uav, "x") && numel(uav.x) >= max(idx)
    value = uav.x(idx);
else
    value = default_value(:);
end
end

function accel = get_command_accel(command)
if isstruct(command) && isfield(command, "accel_cmd")
    accel = command.accel_cmd(:);
elseif isnumeric(command) && numel(command) >= 3
    accel = command(1:3);
else
    accel = zeros(3, 1);
end
end

function yaw = get_command_yaw(command)
if isstruct(command) && isfield(command, "yaw_cmd")
    yaw = command.yaw_cmd;
else
    yaw = 0.0;
end
end

function wind_velocity = get_wind_velocity(wind)
if isstruct(wind) && isfield(wind, "velocity_mps")
    wind_velocity = wind.velocity_mps(:);
elseif isnumeric(wind) && numel(wind) >= 3
    wind_velocity = wind(1:3);
else
    wind_velocity = [0; 0; 0];
end
end

function mass = get_mass(params)
if isfield(params, "m")
    mass = params.m;
elseif isfield(params, "mass")
    mass = params.mass;
else
    mass = 1.30;
end
end

function gravity = get_gravity(params)
if isfield(params, "g")
    gravity = params.g;
else
    gravity = 9.81;
end
end

function J = get_inertia(params)
if isfield(params, "J")
    J = params.J;
elseif isfield(params, "I")
    J = params.I;
else
    J = diag([0.0123, 0.0123, 0.0224]);
end
end

function R = quat_to_dcm_local(q)
q = q(:) ./ max(norm(q), eps);
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1 - 2*(y^2 + z^2), 2*(x*y - z*w), 2*(x*z + y*w); ...
     2*(x*y + z*w), 1 - 2*(x^2 + z^2), 2*(y*z - x*w); ...
     2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x^2 + y^2)];
end

function q = dcm_to_quat_local(R)
tr = trace(R);
if tr > 0
    S = sqrt(tr + 1.0) * 2;
    q = [0.25 * S; (R(3,2) - R(2,3)) / S; (R(1,3) - R(3,1)) / S; (R(2,1) - R(1,2)) / S];
elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
    S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2;
    q = [(R(3,2) - R(2,3)) / S; 0.25 * S; (R(1,2) + R(2,1)) / S; (R(1,3) + R(3,1)) / S];
elseif R(2,2) > R(3,3)
    S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2;
    q = [(R(1,3) - R(3,1)) / S; (R(1,2) + R(2,1)) / S; 0.25 * S; (R(2,3) + R(3,2)) / S];
else
    S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2;
    q = [(R(2,1) - R(1,2)) / S; (R(1,3) + R(3,1)) / S; (R(2,3) + R(3,2)) / S; 0.25 * S];
end
q = q ./ max(norm(q), eps);
end
