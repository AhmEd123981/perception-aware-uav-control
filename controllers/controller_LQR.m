function ctrl = controller_LQR(params, dt, options)
    %% LQR Controller Initialization
    % Full-state feedback with integral action on Z-axis.

    if nargin < 2 || isempty(dt), dt = 0.01; end
    if nargin < 3 || isempty(options), options = struct(); end
    verbose = controller_options_verbose(options);

    ctrl = struct();
    ctrl.type = 'LQR';
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
        % Get both continuous and discrete linearized matrices
        [A_c, B_c, A_d, B_d] = get_linearized_matrices(params, dt);
        ctrl.A_d = A_d;
        ctrl.B_d = B_d;

        % LQR weighting matrices
        Q = diag([2, 2, 5,  8, 8, 12,  5, 5, 2,  3, 3, 2]);
        R = diag([2, 80, 80, 100]);

        % Use continuous-time matrices for gain design
        try
            ctrl.K = lqr(A_c, B_c, Q, R);
        catch
            X = care(A_c, B_c, Q, R);
            ctrl.K = R \ B_c' * X;
        end

        % Verify discrete-time closed-loop stability
        cl_eigs_d = eig(A_d - B_d * ctrl.K);
        max_abs_eig = max(abs(cl_eigs_d));
        if max_abs_eig >= 1.0
            if verbose
                warning('LQR: continuous gain gives discrete |eig|=%.4f >= 1, switching to dlqr', max_abs_eig);
            end
            ctrl.K = dlqr(A_d, B_d, Q, R);
            cl_eigs_d = eig(A_d - B_d * ctrl.K);
            max_abs_eig = max(abs(cl_eigs_d));
        end

        % Integral action for zero steady-state error (Z-axis only)
        ctrl.use_integral = true;
        ctrl.Ki_pos       = [0.4; 0.4; 0.8];
        ctrl.integral_pos = zeros(3, 1);
        ctrl.integral_limit = 2.0;

        ctrl.condition_number = cond(ctrl.K);

        if verbose
            fprintf('LQR Controller initialized:\n');
            fprintf('   - Gain matrix condition number : %.2f\n', ctrl.condition_number);
            fprintf('   - Discrete closed-loop max|eig|: %.4f (must be < 1)\n', max_abs_eig);
        end

    catch ME
        error('LQR controller initialization failed: %s', ME.message);
    end
end

function [A_c, B_c, A_d, B_d] = get_linearized_matrices(params, dt)
    A_c = zeros(12, 12);
    A_c(1:3, 4:6)   = eye(3);
    A_c(4,  8)      =  params.g;
    A_c(5,  7)      = -params.g;
    A_c(7:9, 10:12) = eye(3);

    B_c = zeros(12, 4);
    B_c(6,  1) = 1 / params.m;
    B_c(10, 2) = 1 / params.Jxx;
    B_c(11, 3) = 1 / params.Jyy;
    B_c(12, 4) = 1 / params.Jzz;

    sys_c = ss(A_c, B_c, [], []);
    sys_d = c2d(sys_c, dt);
    A_d = sys_d.A;
    B_d = sys_d.B;
end
