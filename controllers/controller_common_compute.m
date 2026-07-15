function [command, ctrl] = controller_common_compute(uav, x_ref, ctrl, params, math)
%CONTROLLER_COMMON_COMPUTE Controller-specific position command.
%   Implements the per-step API [u, ctrl] used by the perception integration.

if nargin < 5
    math = struct(); %#ok<NASGU>
end

dt = ctrl.dt;
position = get_position(uav, x_ref);
velocity = get_velocity(uav, x_ref);
ref_pos = x_ref(1:3)';
ref_vel = x_ref(4:6)';

error_pos = ref_pos - position;
error_vel = ref_vel - velocity;
ctrl.integral_error = clamp(ctrl.integral_error + error_pos * dt, -ctrl.integral_limit, ctrl.integral_limit);

base_accel = ctrl.kp .* error_pos + ctrl.kd .* error_vel + ctrl.ki .* ctrl.integral_error;

switch char(ctrl.kind)
    case "hinfinity"
        base_accel = base_accel + ctrl.robust_bias .* tanh(error_vel ./ 0.25);
    case "mpc"
        base_accel = base_accel + preview_damping(ctrl, x_ref, velocity);
    case "sliding_mode"
        surface = error_vel + ctrl.lambda_pos .* error_pos;
        switching = ctrl.k_switch .* sat(surface ./ ctrl.phi);
        base_accel = base_accel + switching;
end

base_accel = saturate_vector(base_accel, ctrl.max_accel);
ctrl.filtered_accel = ctrl.response_alpha .* base_accel + (1 - ctrl.response_alpha) .* ctrl.filtered_accel;
ctrl.prev_error = error_pos;
ctrl.step_count = ctrl.step_count + 1;

mass = get_mass(params);
g = get_gravity(params);
command = struct();
command.controller = ctrl.name;
command.accel_cmd = ctrl.filtered_accel;
command.thrust = mass * max(0.0, g + ctrl.filtered_accel(3));
command.yaw_cmd = 0.0;
command.body_rate_cmd = zeros(3, 1);
command.kind = ctrl.kind;
end

function damping = preview_damping(ctrl, x_ref, velocity)
if ctrl.preview_steps <= 0
    damping = zeros(3, 1);
    return;
end
preview_vel = x_ref(4:6)';
damping = 0.35 .* (preview_vel - velocity);
end

function y = sat(x)
y = max(-1, min(1, x));
end

function out = saturate_vector(v, max_norm)
n = norm(v);
if n > max_norm
    out = v * (max_norm / n);
else
    out = v;
end
end

function value = clamp(value, lo, hi)
value = min(max(value, lo), hi);
end

function position = get_position(uav, x_ref)
if isfield(uav, "position")
    position = uav.position(:);
elseif isfield(uav, "pos")
    position = uav.pos(:);
elseif isfield(uav, "x") && numel(uav.x) >= 3
    position = uav.x(1:3);
else
    position = x_ref(1:3)';
end
end

function velocity = get_velocity(uav, x_ref)
if isfield(uav, "velocity")
    velocity = uav.velocity(:);
elseif isfield(uav, "vel")
    velocity = uav.vel(:);
elseif isfield(uav, "x") && numel(uav.x) >= 6
    velocity = uav.x(4:6);
else
    velocity = x_ref(4:6)';
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

function g = get_gravity(params)
if isfield(params, "g")
    g = params.g;
else
    g = 9.81;
end
end
