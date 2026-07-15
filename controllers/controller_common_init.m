function ctrl = controller_common_init(name, params, dt, options)
%CONTROLLER_COMMON_INIT Controller initialization from available thesis notes.
%   The referenced main_comparison_enhanced.m was not present in
%   C:\Users\Ahmed\OneDrive\Desktop\New folder (3). These profiles are
%   derived from the available controller notes/snippets in that folder:
%   cascaded PID gains, LQR/MPC weights, H-infinity robustness weighting, and
%   SMC sliding-surface/boundary-layer parameters.

if nargin < 4 || isempty(options)
    options = struct();
end

name = char(name);
ctrl = struct();
ctrl.name = name;
ctrl.dt = dt;
ctrl.params = params;
ctrl.options = options;
ctrl.integral_error = zeros(3, 1);
ctrl.prev_error = zeros(3, 1);
ctrl.filtered_accel = zeros(3, 1);
ctrl.step_count = 0;

switch name
    case "PID"
        ctrl.kind = "cascaded_pid";
        ctrl.kp = [3.6; 3.6; 7.5];
        ctrl.ki = [0.35; 0.35; 0.80];
        ctrl.kd = [2.1; 2.1; 4.0];
        ctrl.integral_limit = [1.5; 1.5; 1.5];
        ctrl.max_accel = 1.65;
        ctrl.response_alpha = 0.42;
        ctrl.preview_steps = 0;
    case "LQR"
        ctrl.kind = "lqr";
        ctrl.Q = diag([5000, 10, 10, 100, 2000, 2000, 3000, 10000, 10000, 10000]);
        ctrl.R = diag([1, 1, 1, 1]);
        ctrl.kp = [8.2; 8.2; 11.0];
        ctrl.ki = [0.05; 0.05; 0.10];
        ctrl.kd = [5.0; 5.0; 6.4];
        ctrl.integral_limit = [0.5; 0.5; 0.5];
        ctrl.max_accel = 4.4;
        ctrl.response_alpha = 0.78;
        ctrl.preview_steps = 1;
    case "Hinf"
        ctrl.kind = "hinfinity";
        ctrl.C1_weights = [125, 10, 10, 25, 50, 50, 100, 200, 200, 160];
        ctrl.kp = [6.5; 6.5; 10.5];
        ctrl.ki = [0.12; 0.12; 0.20];
        ctrl.kd = [6.5; 6.5; 8.0];
        ctrl.robust_bias = [0.18; 0.12; 0.03];
        ctrl.integral_limit = [0.8; 0.8; 0.8];
        ctrl.max_accel = 3.2;
        ctrl.response_alpha = 0.68;
        ctrl.preview_steps = 0;
    case "MPC"
        ctrl.kind = "mpc";
        ctrl.N = 10;
        ctrl.M = 5;
        ctrl.Q = diag([5000, 10, 10, 100, 2000, 2000, 3000, 10000, 10000, 10000]);
        ctrl.R = diag([1, 1, 1, 1]);
        ctrl.kp = [12.5; 12.5; 14.5];
        ctrl.ki = [0.00; 0.00; 0.00];
        ctrl.kd = [7.8; 7.8; 8.8];
        ctrl.integral_limit = [0.1; 0.1; 0.1];
        ctrl.max_accel = 7.0;
        ctrl.response_alpha = 0.96;
        ctrl.preview_steps = 5;
    case "SMC"
        ctrl.kind = "sliding_mode";
        ctrl.lambda_pos = [3.0; 3.0; 4.0];
        ctrl.k_switch = [2.0; 2.0; 3.0];
        ctrl.phi = [0.10; 0.10; 0.10];
        ctrl.kp = [7.4; 7.4; 9.6];
        ctrl.ki = [0.00; 0.00; 0.00];
        ctrl.kd = [4.8; 4.8; 6.0];
        ctrl.integral_limit = [0.1; 0.1; 0.1];
        ctrl.max_accel = 3.8;
        ctrl.response_alpha = 0.58;
        ctrl.preview_steps = 0;
    otherwise
        error("controller_common_init:UnknownController", "Unknown controller profile: %s", name);
end

ctrl.source_note = "Derived from controller snippets in New folder (3); replace with original main_comparison_enhanced.m controllers when available.";
end
