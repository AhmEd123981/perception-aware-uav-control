function ctrl = controller_SMC(params, dt, options)
    %% Sliding Mode Controller (SMC) for Agricultural Quadrotor
    % Uses boundary-layer chattering reduction.
    % Observer is DISABLED: SMC's inherent robustness handles
    % disturbances and model uncertainty without needing an observer.
    % The observer was differentiating noisy velocity measurements,
    % producing ~5 m/s^2 noise that destabilised the controller.

    if nargin < 2 || isempty(dt), dt = 0.01; end
    if nargin < 3 || isempty(options), options = struct(); end
    verbose = controller_options_verbose(options);

    ctrl = struct();
    ctrl.type = 'SMC';
    ctrl.params = params;
    ctrl.dt = dt;
    ctrl.verbose = verbose;
    ctrl.use_trajectory_feedforward = true;
    ctrl.ff_acc_xy_limit = 4.0;
    ctrl.ff_acc_z_limit = 3.0;
    ctrl.ff_tilt_limit = 25*pi/180;
    ctrl.ff_filter_alpha = exp(-12 * dt);
    ctrl.ff_min_thrust = 0.20 * params.m * params.g;
    ctrl.ff_max_thrust = 2.00 * params.m * params.g;

    try
        % Sliding surface design parameters
        % s = e_dot + lambda * e
        ctrl.lambda_pos = diag([1.6, 1.6, 2.1]);    % Position sliding surface
        ctrl.lambda_att = diag([4.0, 4.0, 2.8]);    % Attitude sliding surface

        % Switching gains (reaching law)
        ctrl.k_pos = diag([1.4, 1.4, 2.2]);         % Position switching gains
        ctrl.k_att = diag([3.0, 3.0, 1.8]);         % Attitude switching gains

        % Boundary layer for chattering reduction
        ctrl.phi_pos = 0.75;
        ctrl.phi_att = 0.35;
        ctrl.max_command_acc_xy = 4.0;
        ctrl.max_command_acc_z = 3.0;
        ctrl.max_command_tilt = 35*pi/180;

        % Adaptive gains — disabled for stability
        ctrl.use_adaptive = false;

        % Disturbance observer — DISABLED
        % SMC is inherently robust to matched disturbances.
        % The observer was the primary source of instability because it
        % amplified sensor noise through velocity differentiation.
        ctrl.use_observer = false;

        % Chattering reduction method
        ctrl.chattering_reduction = 'boundary_layer';
        ctrl.sigmoid_slope = 10.0;

        % Performance monitoring
        ctrl.max_sliding_variable = 0;
        ctrl.avg_switching_freq = 0;
        ctrl.total_control_effort = 0;

        % Integral action — disabled (SMC switching handles SS error)
        ctrl.use_integral = false;

        % Robustness bounds (for documentation)
        ctrl.uncertainty_bound = struct();
        ctrl.uncertainty_bound.mass_variation = 0.3;
        ctrl.uncertainty_bound.inertia_variation = 0.25;
        ctrl.uncertainty_bound.external_disturbance = 2.0;

        if verbose
            fprintf('SMC Controller initialized:\n');
            fprintf('   - Chattering reduction: %s (phi=%.2f)\n', ...
                ctrl.chattering_reduction, ctrl.phi_pos);
            fprintf('   - Observer: Disabled (inherent robustness)\n');
        end

    catch ME
        error('SMC controller initialization failed: %s', ME.message);
    end
end
