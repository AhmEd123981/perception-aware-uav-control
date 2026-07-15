function ctrl = controller_MPC(params, dt, options)
    %% Model Predictive Controller (MPC)
    % Batch QP with box constraints and warm starting.

    if nargin < 2 || isempty(dt), dt = 0.01; end
    if nargin < 3 || isempty(options), options = struct(); end
    verbose = controller_options_verbose(options);

    ctrl = struct();
    ctrl.type = 'MPC';
    ctrl.params = params;
    ctrl.dt = dt;
    ctrl.verbose = verbose;
    ctrl.use_trajectory_feedforward = true;
    ctrl.ff_acc_xy_limit = 4.0;
    ctrl.ff_acc_z_limit = 3.0;
    ctrl.ff_tilt_limit = 25*pi/180;
    ctrl.ff_filter_alpha = exp(-12 * dt);
    ctrl.ff_thrust_blend = 1.0;
    ctrl.ff_min_thrust = 0.20 * params.m * params.g;
    ctrl.ff_max_thrust = 2.00 * params.m * params.g;

    try
        % Prediction horizon
        ctrl.N = 20;
        ctrl.N_control = 15;

        % Hover reference
        ctrl.u_hover = params.m * params.g;
        ctrl.x_hover = [0; 0; 4; zeros(9,1)];

        % Cost weights
        % States: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        ctrl.Q = diag([80, 80, 120, ...
                       30, 30, 45, ...
                       25, 25, 8, ...
                       5,  5,  3]);

        % Control weights: [thrust, tau_x, tau_y, tau_z]
        ctrl.R = diag([0.04, 0.18, 0.18, 0.5]);

        % Terminal cost
        ctrl.QN = ctrl.Q * 20;

        % Decision variable is DEVIATION from hover.
        % Constraints widened to match physical limits so MPC has
        % enough authority under parameter variations.
        thrust_hover = params.m * params.g;
        torque_lim = params.torque_max;
        ctrl.u_min = [-0.7 * thrust_hover; -torque_lim; -torque_lim; -torque_lim];
        ctrl.u_max = [ 0.7 * thrust_hover;  torque_lim;  torque_lim;  torque_lim];

        % State constraints
        ctrl.x_min = [-50; -50; 0.5; -10; -10; -5; -pi/3; -pi/3; -pi; -5; -5; -3];
        ctrl.x_max = [50; 50; 10; 10; 10; 5; pi/3; pi/3; pi; 5; 5; 3];

        % Get discrete linearized model
        [ctrl.A, ctrl.B] = get_mpc_linearized_model(params, dt);

        n = size(ctrl.A, 1);
        m = size(ctrl.B, 2);
        N = ctrl.N;

        % Verify controllability
        Ctrb = ctrb(ctrl.A, ctrl.B);
        if rank(Ctrb) < n
            if verbose
                warning('MPC: system is not fully controllable');
            end
        end

        % Precompute prediction matrices
        Sx = zeros(n*(N+1), n);
        A_pow = eye(n);
        for k = 0:N
            Sx(k*n+1:(k+1)*n, :) = A_pow;
            A_pow = ctrl.A * A_pow;
        end

        Su = zeros(n*(N+1), m*N);
        for k = 1:N
            for j = 1:k
                power = k - j;
                if power == 0
                    AB = ctrl.B;
                else
                    AB = ctrl.A^power * ctrl.B;
                end
                Su(k*n+1:(k+1)*n, (j-1)*m+1:j*m) = AB;
            end
        end

        Q_blk = blkdiag(kron(eye(N), ctrl.Q), ctrl.QN);
        R_blk = kron(eye(N), ctrl.R);

        ctrl.H_qp  = Su' * Q_blk * Su + R_blk;
        ctrl.H_qp  = (ctrl.H_qp + ctrl.H_qp') / 2;
        ctrl.Sx    = Sx;
        ctrl.Su    = Su;
        ctrl.Q_blk = Q_blk;

        ctrl.u_prev = zeros(m*N, 1);

        % Performance monitoring
        ctrl.solve_times      = [];
        ctrl.iterations       = [];
        ctrl.feasible_solutions = 0;
        ctrl.total_calls      = 0;

        if verbose
            fprintf('MPC Controller initialized:\n');
            fprintf('   - Prediction horizon : %d\n', ctrl.N);
            fprintf('   - Control horizon    : %d\n', ctrl.N_control);
            fprintf('   - System rank        : %d/%d\n', rank(Ctrb), n);
            fprintf('   - H_qp condition no. : %.1f\n', cond(ctrl.H_qp));
        end

    catch ME
        error('MPC controller initialization failed: %s', ME.message);
    end
end

function [A, B] = get_mpc_linearized_model(params, dt)
    A_c = zeros(12, 12);
    A_c(1:3, 4:6) = eye(3);
    A_c(4, 8) = params.g;
    A_c(5, 7) = -params.g;
    A_c(7:9, 10:12) = eye(3);

    B_c = zeros(12, 4);
    B_c(6, 1) = 1 / params.m;
    B_c(10, 2) = 1 / params.Jxx;
    B_c(11, 3) = 1 / params.Jyy;
    B_c(12, 4) = 1 / params.Jzz;

    sys_c = ss(A_c, B_c, [], []);
    sys_d = c2d(sys_c, dt);
    A = sys_d.A;
    B = sys_d.B;
end
