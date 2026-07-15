function ctrl = controller_PID(params, dt, options)
    %% Enhanced PID Controller Initialization
    % Improved version with anti-windup and derivative filtering
    
    if nargin < 2 || isempty(dt), dt = 0.01; end
    if nargin < 3 || isempty(options), options = struct(); end
    verbose = controller_options_verbose(options);

    ctrl = struct();
    ctrl.type = 'PID';
    ctrl.params = params;
    ctrl.dt = dt;
    ctrl.verbose = verbose;
    ctrl.use_trajectory_feedforward = true;
    ctrl.ff_acc_xy_limit = 4.0;
    ctrl.ff_acc_z_limit = 3.0;
    ctrl.ff_tilt_limit = 25*pi/180;
    ctrl.ff_filter_alpha = exp(-12 * dt);
    ctrl.ff_min_thrust = 0.20 * params.m * params.g;
    ctrl.ff_max_thrust = 1.80 * params.m * params.g;
    
    % Enhanced PID gains (tuned for agricultural UAV)
    ctrl.kp_pos = [3.3; 3.3; 6.5];        % Force-domain position gains [x, y, z]
    ctrl.ki_pos = [0.12; 0.12; 0.25];     % Conservative integral gains
    ctrl.kd_pos = [3.6; 3.6; 5.8];        % Velocity-error damping gains
    
    ctrl.kp_att = [7.0; 7.0; 3.5];        % Attitude gains [roll, pitch, yaw]
    ctrl.ki_att = [0.05; 0.05; 0.02];     % Small attitude integral gains
    ctrl.kd_att = [3.0; 3.0; 1.8];        % Angular-rate damping gains
    
    % Anti-windup limits
    ctrl.integral_limit_pos = [2.0; 2.0; 3.0];
    ctrl.integral_limit_att = [0.5; 0.5; 0.3];
    
    % Derivative filtering
    ctrl.lpf_cutoff = 20.0;                % Low-pass filter cutoff [rad/s]
    ctrl.alpha = exp(-ctrl.lpf_cutoff * ctrl.dt); % Filter coefficient
    
    % Initialize states
    ctrl.integral_pos = zeros(3, 1);
    ctrl.integral_att = zeros(3, 1);
    ctrl.prev_error_pos = zeros(3, 1);
    ctrl.prev_error_att = zeros(3, 1);
    ctrl.filtered_deriv_pos = zeros(3, 1);
    ctrl.filtered_deriv_att = zeros(3, 1);
    ctrl.filtered_vel_error = zeros(3, 1);
    ctrl.filtered_omega_error = zeros(3, 1);
    
    if verbose
        fprintf('✅ Enhanced PID Controller initialized with anti-windup and filtering\n');
    end
end
